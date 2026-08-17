#!/usr/bin/env python3
"""ship_to_drive.py — runs on the Colab VM, fully headless.

Waits for the in-progress fine-tune (daily_finetune.py, log /content/train11.log)
to finish, then ships the resulting LoRA adapter + config + metrics to Google
Drive so the result survives the free-tier session recycle even if the local
machine that launched it is powered off.

Auth: vendored gdrive.py + gcloud ADC `authorized_user` refresh token — a
server-side OAuth refresh (no browser, no interaction). No secrets are ever
printed; credentials live in /root/.config/gcloud/application_default_credentials.json.
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

OUT_DIR = os.environ.get("OUT_DIR", "/content/out")
TRAIN_LOG = os.environ.get("TRAIN_LOG", "/content/train11.log")
# Deployment-specific values come from env (see README); no hardcoded IDs in the repo.
RESULTS_FOLDER = os.environ.get("DRIVE_RESULTS", "")
GDRIVE = "/content/gdrive.py"
QUOTA_PROJECT = os.environ.get("GDRIVE_QUOTA_PROJECT", "")  # empty -> gdrive auto-discovers

ENV = dict(os.environ)
if QUOTA_PROJECT:
    ENV["GDRIVE_QUOTA_PROJECT"] = QUOTA_PROJECT


def log(msg):
    print(f"[ship {datetime.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


def run_gdrive(args, timeout=900):
    cmd = [sys.executable, GDRIVE] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=timeout)
    except Exception as e:
        log(f"gdrive {' '.join(args)} raised: {e}")
        return -1, "", ""
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    log(f"gdrive {' '.join(args)} -> rc={r.returncode} out={out[:160]} err={err[:160]}")
    return r.returncode, out, err


def training_done():
    adapter = pathlib.Path(OUT_DIR) / "adapter_model.safetensors"
    metrics = pathlib.Path(OUT_DIR) / "metrics.json"
    if adapter.exists() and metrics.exists():
        return True
    if pathlib.Path(TRAIN_LOG).exists():
        txt = pathlib.Path(TRAIN_LOG).read_text(errors="ignore")
        if "[RESULT]" in txt or "EXIT " in txt:
            return True
    return False


def main():
    if not RESULTS_FOLDER:
        log("DRIVE_RESULTS env not set — nothing to ship to")
        sys.exit(1)
    log("waiting for training to complete...")
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        if training_done():
            break
        time.sleep(45)
    else:
        log("TIMEOUT (3h) waiting for training; session likely recycled")
        sys.exit(2)

    time.sleep(20)  # allow trainer to flush final files

    adapter = pathlib.Path(OUT_DIR) / "adapter_model.safetensors"
    metrics = pathlib.Path(OUT_DIR) / "metrics.json"
    if not adapter.exists():
        log(f"training finished but no adapter at {adapter} — checking log tail")
        if pathlib.Path(TRAIN_LOG).exists():
            txt = pathlib.Path(TRAIN_LOG).read_text(errors="ignore")
            for line in txt.splitlines()[-5:]:
                log("  log: " + line[:200])
        sys.exit(3)

    try:
        size = adapter.stat().st_size
    except Exception:
        size = 0
    log(f"adapter {adapter} ({size} bytes); metrics={metrics.exists()}")

    try:
        import googleapiclient  # noqa: F401
        import google.auth  # noqa: F401
    except Exception:
        log("installing google-api-python-client + google-auth")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-input",
             "google-api-python-client", "google-auth"],
            check=False,
        )

    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    folder_name = f"manual_{stamp}"
    rc, out, _ = run_gdrive(["mkdir", folder_name, "--parent", RESULTS_FOLDER])
    folder_id = None
    try:
        folder_id = json.loads(out).get("id")
    except Exception:
        pass
    if not folder_id:
        log(f"mkdir failed (rc={rc}) — aborting")
        sys.exit(4)

    uploaded = []
    for p in sorted(pathlib.Path(OUT_DIR).iterdir()):
        if p.is_file():
            rc, _, _ = run_gdrive(["upload", str(p), "--parent", folder_id])
            uploaded.append((p.name, rc == 0))
    ok = all(v for _, v in uploaded)
    log(f"uploaded {sum(v for _, v in uploaded)}/{len(uploaded)} files -> results/{folder_name}/ ({folder_id})")
    for name, v in uploaded:
        if not v:
            log(f"  FAILED upload: {name}")
    log("DONE")
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
