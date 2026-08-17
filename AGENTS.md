# AGENTS.md — soup-daily-finetune

Guidance for AI coding agents working in this repo.

## What this is

A headless daily LoRA fine-tune of Llama-3.1-8B-Instruct on financial data, run on
a free Google Colab T4 and orchestrated by GitHub Actions, with Google Drive as the
persistent model store. No local machine is needed at run time.

## Architecture

- `run_daily.py` — orchestrator (GitHub Actions runner). Mints Colab access token
  from a refresh token (server-side OAuth), writes `~/.config/colab-cli/token.json`
  + gcloud ADC, drives `colab.py` + `gdrive.py` via subprocess, polls the VM log,
  downloads the adapter, uploads to Drive, stops the session.
- `daily_finetune.py` — the training script (runs on the Colab VM). Installs the
  known-good stack, pulls the current adapter from Drive (`gdown --folder`),
  continues fine-tuning (peft + trl + bitsandbytes), saves adapter + metrics.
- `colab.py` — vendored Colab CLI v3.2 (wraps `google-colab-cli`; adds
  `exec_detach`, token auto-refresh, retries).
- `gdrive.py` — vendored Google Drive CLI (reads gcloud ADC via `GDRIVE_ADC`).
- `.github/workflows/daily-finetune.yml` — cron + manual trigger.

## Hard rules

1. **Never commit credentials.** All secrets are GitHub-encrypted and passed via
   env vars. `.gitignore` excludes every credential path (`*.json`, token files).
   If you add code, grep for `client_secret`, `refresh_token`, `access_token`,
   `764086051850`, email addresses, or `/home/` paths before committing.
2. **Colab free tier is ephemeral.** Assume the VM is recycled after ~2–3 h and
   `/content` is wiped. The persisted artifact is the adapter on Drive, not
   anything on the VM.
3. **Don't layer-stream 8B on a T4.** Resident NF4 QLoRA fits (9.4 GB of 15.36 GB)
   and is ~1.6–2.6× faster. Streaming is for <4 GB cards or >14B models.
4. **The bf16 cast is load-bearing.** On a T4, peft 0.19.x creates bf16 LoRA
   adapters on a bf16 base checkpoint; the fp16 GradScaler crashes on bf16 grads
   (`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`).
   `daily_finetune.py` casts `lora_` params to fp32 before training. Don't remove it.

## Known-good dependency pins (Colab ships wrong versions)

Colab T4 image ships `transformers 5.13.1` + `torch 2.11.0+cu128` + `peft 0.19.1`.
The script installs the pins Soup verified:
`transformers>=4.46,<5.0`, `trl>=0.14,<0.29`, `bitsandbytes>=0.41`,
`accelerate>=0.25`. Also uninstalls `torchao` (Colab's stale 0.10.0 makes peft raise).

## Drive layout

- `soup-finetune/adapter_in/` — the "latest" adapter (shared anyone-with-link so
  `gdown --folder` can pull it without auth). The orchestrator trashes old files and
  uploads new ones each day (trash, not permanent — recoverable).
- `soup-finetune/results/<date>/` — dated archive of each day's adapter + metrics.

## Pitfalls learned the hard way

- **`colab.py upload` of large files (>~100 MB) fails with "Network error"** — the
  contents API caps large uploads. Workaround: put the adapter on Drive and pull it
  with `gdown --folder` on the VM instead of uploading it.
- **`gdown --folder` needs the folder shared anyone-with-link** (`gdrive share <id>
  --type anyone --role reader`), and it downloads all non-trashed files.
- **Refresh tokens are long-lived** (gcloud client 764086051850 is production, not
  testing), so the headless OAuth refresh in `run_daily.py` works indefinitely
  until the user revokes access.
- **Colab daily GPU quota** is ~3–4 T4 sessions/account. A stuck/failed run that
  leaves a session idle wastes quota — `run_daily.py` always calls `stop` in the
  failure paths too.
