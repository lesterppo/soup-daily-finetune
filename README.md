# soup-daily-finetune

A **fully automated, headless daily LoRA fine-tune** of `Llama-3.1-8B-Instruct` on
financial-domain data — running on a **free Google Colab T4** and persisting every
model to **Google Drive**, orchestrated entirely by **GitHub Actions**. The machine
that first built this pipeline is **not required** at run time.

```
GitHub Actions (daily cron) ──► Google Colab T4 (train) ──► Google Drive (model store)
```

## What it does, each day

1. A GitHub Actions workflow fires on a cron schedule (10:00 PM HKT).
2. It mints a Colab access token from a stored refresh token (server-side OAuth,
   no browser), and loads Drive credentials the same way.
3. It spins up a fresh free-Colab T4 session and runs `daily_finetune.py`.
4. The VM pulls the **current** LoRA adapter from Google Drive (`gdown --folder`),
   continues fine-tuning it on a fresh `gbharti/finance-alpaca` subset (seed =
   today's date, so each day trains on a different shuffle), and saves the updated
   adapter.
5. The orchestrator downloads the result and uploads it to Drive twice: a dated
   archive (`results/<date>/`) and the `adapter_in/` folder that tomorrow's run
   continues from.

The result is a **continually improving** financial-domain LoRA adapter, with a
dated archive of every daily checkpoint on Drive.

## Architecture

| Component | File | Where it runs |
|---|---|---|
| Orchestrator | `run_daily.py` | GitHub Actions runner (headless) |
| Training script | `daily_finetune.py` | free Colab T4 (GPU) |
| Colab driver | `colab.py` (v3.2) | GitHub Actions runner |
| Drive client | `gdrive.py` | GitHub Actions runner |
| Scheduler | `.github/workflows/daily-finetune.yml` | GitHub Actions |

### Why not the Soup CLI for the daily loop?

This pipeline was born from [Soup](https://github.com/MakazhanAlpamys/Soup) — the
fine-tuning config below replicates Soup's proven "v2" recipe exactly. The daily
loop drives the underlying stack (peft + trl + bitsandbytes) directly for three
reasons:

1. **"Continue from an adapter" isn't in Soup's CLI.** `soup train --resume`
   resumes a full checkpoint (optimizer + scheduler state); loading an existing
   adapter as init weights (`PeftModel.from_pretrained`) is not exposed. A daily
   incremental loop needs exactly that.
2. **Headless reliability.** `soup train` prompts interactively; a script that
   must run unattended on a recycled VM wants zero prompts and programmatic control.
3. **The pre-Ampere bf16 fix (Soup PR #429) is baked in** — on a T4, peft 0.19.x
   creates bf16 LoRA adapters on a bf16 base checkpoint, and the fp16 GradScaler
   can't unscale bf16 gradients. The script casts `lora_` params to fp32 before
   the optimizer sees them.

### Training config (replicates Soup "v2")

- **base**: `NousResearch/Meta-Llama-3.1-8B-Instruct` (NF4 QLoRA, 4-bit)
- **LoRA**: r=16, α=32, all 7 linear layers (q,k,v,o,gate,up,down)
- **batch** 4, **seq** 512, **lr** 2e-4 (cosine, warmup 3%), 1 epoch, fp16
- **resident** QLoRA (NOT layer streaming — on a 16 GB T4 resident batch-4 is
  ~135 tok/s vs ~53 tok/s streamed; streaming is for <4 GB cards or >14B models)

## Setup (one-time)

### 1. Prerequisites (already done on the authoring machine)

- A free Colab account with OAuth (`colab-cli` token at `~/.config/colab-cli/token.json`).
- A gcloud ADC credential with the `colaboratory` + `drive.file` scopes
  (`gcloud auth application-default login`).
- Google Drive folders: `soup-finetune/adapter_in/` (shared anyone-with-link,
  holds the adapter the VM pulls) and `soup-finetune/results/` (dated archives).

### 2. GitHub secrets

Set these on the repo (never commit them):

| Secret | Value |
|---|---|
| `COLAB_CLIENT_ID` | OAuth client id |
| `COLAB_CLIENT_SECRET` | OAuth client secret |
| `COLAB_REFRESH_TOKEN` | long-lived refresh token (`colaboratory` + `drive.file` scope) |
| `GDRIVE_ADC` | full gcloud ADC JSON (`{"type":"authorized_user","client_id":...,"client_secret":...,"refresh_token":...}`) |
| `DRIVE_ADAPTER_IN` | Drive folder id for `adapter_in/` (shared anyone-with-link) |
| `DRIVE_RESULTS` | Drive folder id for `results/` |

```bash
gh secret set COLAB_CLIENT_ID -b"..."   # etc.
```

### 3. Seed the loop

Upload a starting adapter to `adapter_in/` (or run once and it will create one
from scratch — `daily_finetune.py` starts a fresh LoRA when the folder is empty).

### 4. Test

```bash
gh workflow run daily-finetune.yml
```

## Manual run (on a machine with Colab + Drive access)

```bash
pip install -r requirements.txt
python run_daily.py
```

## Notes & limits

- **Free-Colab T4** sessions recycle after ~2–3 h and the ephemeral disk is wiped;
  the base model (~15 GB) re-downloads each run. Training is sized (5000 rows,
  ~70–90 min) to finish inside one session.
- **Daily quota** is ~3–4 T4 sessions per account; a single run uses one.
- The adapter is the persisted artifact; the base model always comes from
  `NousResearch` (ungated HF mirror), not Drive.

## Privacy

No credentials, tokens, or secrets live in this repo — they are GitHub-encrypted
secrets referenced by the workflow, and `.gitignore` excludes every credential
path.
