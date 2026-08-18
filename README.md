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
4. The VM **mounts Google Drive itself** (vendored `gdrive.py` + the gcloud ADC
   JSON, uploaded alongside the training script — headless, no browser) and
   resumes from the **most-updated adapter on Drive**: newest step checkpoint
   under `results/checkpoints/` (mid-run progress), falling back to the newest
   dated archive `results/<date>/` (a completed fine-tuned model), then to the
   shared `adapter_in/` folder via `gdown --folder` on the first run.
5. Training continues fine-tuning on a fresh `gbharti/finance-alpaca` subset
   (seed = today's date, so each day trains on a different shuffle).
6. **Every `--save-steps` training steps, the VM pushes a checkpoint to Drive**
   (`results/checkpoints/<run-id>/step-<n>-...`, keeping the newest 2). This is
   the timeout/recycle safety net: a free-Colab session is recycled after
   ~2-3 h and the ephemeral disk is wiped, but the latest checkpoint is already
   on Drive, so a killed run loses at most `save_steps` of work.
7. When training ends (naturally or via the `--max-minutes` budget), the VM
   archives the final adapter under `results/<date>/`, updates the `adapter_in/`
   continuity pointer, and reports `[RESULT]`. The orchestrator downloads the
   result for the GH artifact tab and stops the session. On runner timeout it
   pulls the newest Drive checkpoint into `adapter_in/` so the next day resumes
   from real progress.

The result is a **continually improving** financial-domain LoRA adapter, with a
dated archive of every daily checkpoint on Drive — resilient to GitHub Actions
timeouts, runner death, and Colab session recycling.

## Architecture

| Component | File | Where it runs |
|---|---|---|
| Orchestrator | `run_daily.py` | GitHub Actions runner (headless) |
| Training script | `daily_finetune.py` | free Colab T4 (GPU) |
| **VM-side auto-ship** | `ship_to_drive.py` | free Colab T4 (GPU) |
| Colab driver | `colab.py` (v3.2) | GitHub Actions runner |
| Drive client | `gdrive.py` | GitHub Actions runner **and** Colab VM |
| Scheduler | `.github/workflows/daily-finetune.yml` | GitHub Actions |

### VM-side Drive checkpointing (the timeout/recycle safety net)

`daily_finetune.py` mounts Google Drive **on the VM** (vendored `gdrive.py` +
the ADC JSON the orchestrator uploads; headless refresh-token OAuth, no
browser). A `TrainerCallback` pushes a checkpoint to
`results/checkpoints/<run-id>/step-<n>-adapter_model.safetensors` after every
`--save-steps` training steps and prunes to the newest 2. `<run-id>` is unique
per run (`YYYYMMDD-HHMMSS`), so same-day re-runs never collide on step numbers.
On startup it scans all run folders under `results/checkpoints/` and resumes
from the globally newest step (falling back to the newest dated archive if no
checkpoints exist). `--max-minutes` (default 100) makes the VM stop cleanly
before the free-tier session recycle window and push a final checkpoint — so
neither a GitHub Actions timeout (`timeout-minutes: 200`) nor a Colab recycle
can lose more than `save_steps` steps of work.

### VM-side auto-ship (`ship_to_drive.py`) — belt and suspenders

The orchestrator (`run_daily.py`) downloads the result and uploads it to Drive.
As a redundant safety net, `ship_to_drive.py` runs **on the VM itself**: it waits
for `daily_finetune.py` to finish, then uploads the adapter + config + metrics
straight to `results/manual_<timestamp>/` on Drive using the same `gdrive.py` +
gcloud-ADC (server-side refresh-token) auth. This guarantees the model is saved
even if the GitHub Actions runner or the machine that launched the session dies
mid-run. `colabctl exec_detach -f ship_to_drive.py --log /content/ship.log` is the
launch pattern; it survives the launching machine being powered off because it is
a detached process on the Colab VM.

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
3. **The T4 needs bf16, not fp16.** On a T4 (sm_75), `fp16` mixed precision uses a
   GradScaler whose `_amp_foreach_non_finite_check_and_unscale_cuda` kernel has no
   bf16 path — and the LoRA gradients come out bf16 regardless of the param dtype —
   so `soup train --fp16` crashes at `clip_grad_norm`. The script uses **bf16**
   mixed precision (the canonical QLoRA-on-T4 recipe): no GradScaler, no crash.

### Training config (replicates Soup "v2")

- **base**: `NousResearch/Meta-Llama-3.1-8B-Instruct` (NF4 QLoRA, 4-bit)
- **LoRA**: r=16, α=32, all 7 linear layers (q,k,v,o,gate,up,down)
- **batch** 4, **seq** 512, **lr** 2e-4 (cosine, warmup 3%), 1 epoch, bf16
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
  the base model (~15 GB) re-downloads each run. Training is sized (2000 rows,
  ~500 steps ≈ 90 min) to finish inside one session.
- **Daily quota** is ~3–4 T4 sessions per account; a single run uses one.
- The adapter is the persisted artifact; the base model always comes from
  `NousResearch` (ungated HF mirror), not Drive.

## Privacy

No credentials, tokens, or secrets live in this repo — they are GitHub-encrypted
secrets referenced by the workflow, and `.gitignore` excludes every credential
path.
