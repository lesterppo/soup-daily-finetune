#!/usr/bin/env python3
"""
daily_finetune.py — daily financial-domain LoRA fine-tune of Llama-3.1-8B-Instruct
on a free Colab T4. Headless, idempotent, self-contained.

The daily loop: load the CURRENT adapter (uploaded alongside this script) and
CONTINUE fine-tuning it on a fresh finance-alpaca subset, then save the updated
adapter + metrics to /content/out/. The orchestrator (GitHub Action, or the local
colabctl driver) handles the Google Drive I/O around it — this script only trains.

Stack (the exact deps Soup wraps, with Soup's known-good pins):
    transformers<5.0   (Colab ships 5.13.1 — must downgrade)
    trl>=0.14,<0.29
    peft, bitsandbytes, accelerate, datasets, safetensors
    torch (Colab's 2.11.0+cu128)

Fixes baked in:
    - uninstall torchao (Colab's stale 0.10.0 makes peft raise)
    - cast bf16 LoRA adapters -> fp32 before the fp16 GradScaler (Soup PR #429)

Config is the proven "v2" recipe: NF4 QLoRA, r=16/alpha=32, all 7 linear layers,
batch 4, seq 512, lr 2e-4, fp16 (T4 has no bf16 hardware), resident (no streaming
— streaming is for <4GB cards; resident batch-4 is ~135 tok/s on a 16GB T4).

Usage (on the VM):
    python daily_finetune.py --rows 5000 --seed 42 \
        --adapter /content/adapter_in --out /content/out
"""
import argparse
import importlib.util
import json
import os
import pathlib
import random
import subprocess
import sys
import time


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=False)


def install_deps():
    """Idempotent install of the known-good training stack."""
    if importlib.util.find_spec("trl") is not None and importlib.util.find_spec("bitsandbytes") is not None:
        # still need to verify transformers < 5.0
        import transformers  # noqa: F401
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    run(f"{sys.executable} -m pip uninstall -q -y torchao")
    run(
        f"{sys.executable} -m pip install -q --no-input "
        f"'transformers>=4.46.0,<5.0.0' 'trl>=0.14.0,<0.29' "
        f"'bitsandbytes>=0.41.0' 'accelerate>=0.25.0' 'peft>=0.7.0' "
        f"'datasets>=2.14.0' 'safetensors'"
    )
    # report versions
    for m in ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"]:
        try:
            mod = __import__(m)
            print(f"[deps] {m} {getattr(mod, '__version__', '?')}", flush=True)
        except Exception as e:
            print(f"[deps] {m} FAILED: {e}", flush=True)


def build_data(n_rows: int, seed: int, out_path: str) -> int:
    from datasets import load_dataset

    ds = load_dataset("gbharti/finance-alpaca", split="train")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    rows = []
    for i in idxs[:n_rows]:
        ex = ds[int(i)]
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not instr or not out:
            continue
        user = f"{instr}\n\n{inp}" if inp else instr
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": out},
                ]
            }
        )
    pathlib.Path(out_path).write_text("\n".join(json.dumps(r) for r in rows))
    print(f"[data] wrote {len(rows)} ChatML rows -> {out_path}", flush=True)
    return len(rows)


def load_model_and_adapter(base: str, adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # standard QLoRA recipe (works on T4)
    )
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
    )
    # NOTE: deliberately NO prepare_model_for_kbit_training() here. Soup's SFT
    # path skips it; calling it casts the frozen bf16 base (embedding/norm/lm_head)
    # to fp32, which then produces bf16 gradients under the fp16 autocast and
    # crashes the fp16 GradScaler on a T4 (see Soup PR #429 / issue #425).

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    if adapter_path and os.path.isdir(adapter_path) and os.path.exists(
        os.path.join(adapter_path, "adapter_model.safetensors")
    ):
        print(f"[adapter] continuing from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print("[adapter] starting fresh LoRA", flush=True)
        model = get_peft_model(model, lora_config)

    # bf16 mixed precision uses NO GradScaler, so no bf16-gradient unscale crash.
    # Trainable LoRA params stay fp32 (master weights); the bf16 autocast handles
    # the forward. (The fp16 GradScaler path crashes on a T4: peft creates bf16
    # adapters on a bf16 base, and `_amp_foreach_non_finite_check_and_unscale_cuda`
    # has no bf16 kernel on sm_75 — Soup PR #429 / issue #425.)
    from collections import Counter
    full = Counter(str(p.dtype) for p in model.parameters())
    trainable = Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)
    print(f"[dtype] ALL params: {dict(full)}", flush=True)
    print(f"[dtype] trainable: {dict(trainable)}", flush=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_trainable}", flush=True)
    return model, tok


def train(model, tok, data_path: str, out_dir: str):
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    ds = load_dataset("json", data_files=data_path, split="train")
    print(f"[data] loaded {len(ds)} rows", flush=True)

    def formatting_func(example):
        return tok.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=1234,
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        formatting_func=formatting_func,
    )

    print("[train] starting", flush=True)
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"[train] done in {dt:.1f}s", flush=True)

    # save adapter
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[save] adapter -> {out_dir}", flush=True)

    # write metrics
    metrics = {
        "train_runtime_s": round(dt, 1),
        "train_samples": len(ds),
        "train_steps": trainer.state.global_step,
        "train_loss": round(float(trainer.state.log_history[-1].get("loss", -1)), 4)
        if trainer.state.log_history else None,
    }
    pathlib.Path(out_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] {json.dumps(metrics)}", flush=True)
    return metrics


def verify_adapter(out_dir: str) -> bool:
    from safetensors.torch import load_file

    ad = pathlib.Path(out_dir) / "adapter_model.safetensors"
    if not ad.exists():
        print(f"[ADAPTER] NOT FOUND at {ad} — training failed", flush=True)
        return False
    tensors = load_file(str(ad))
    live = sum(1 for v in tensors.values() if v.abs().max().item() > 0)
    print(f"[ADAPTER] {len(tensors)} tensors, {live} non-zero -> {ad}", flush=True)
    return live > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", default=None, help="dir with adapter_model.safetensors to continue from")
    ap.add_argument("--adapter-from-drive", default=None,
                    help="Google Drive FOLDER id (shared anyone-with-link) to gdown-download the adapter from")
    ap.add_argument("--out", default="/content/out")
    ap.add_argument("--base", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if not args.skip_install:
        install_deps()

    if args.adapter_from_drive and (not args.adapter or not os.path.isdir(args.adapter)):
        run(f"{sys.executable} -m pip install -q gdown")
        os.makedirs("/content/adapter_in", exist_ok=True)
        r = run(f"gdown --folder {args.adapter_from_drive} -O /content/adapter_in")
        if r.returncode != 0:
            print("[adapter] gdown --folder failed; check the folder is shared anyone-with-link", flush=True)
        args.adapter = "/content/adapter_in"

    build_data(args.rows, args.seed, "/content/finance_train.jsonl")
    model, tok = load_model_and_adapter(args.base, args.adapter)
    metrics = train(model, tok, "/content/finance_train.jsonl", args.out)
    ok = verify_adapter(args.out)
    print(f"\n[RESULT] ok={ok} {json.dumps(metrics)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
