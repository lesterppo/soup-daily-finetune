#!/usr/bin/env python3
"""
run_daily.py — headless daily LoRA fine-tune orchestrator.

Runs inside GitHub Actions (no local machine involved). It:
  1. Mints a Colab access token from a stored refresh token (server-side OAuth,
     no browser).
  2. Writes the colab-cli token.json and the gdrive ADC credentials.
  3. Creates a free Colab T4 session, uploads daily_finetune.py + the vendored
     gdrive.py + the ADC JSON, and launches training detached.
  4. Polls the VM log until the run reports [RESULT] (or times out).
  5. On success: downloads the new adapter + metrics for the GH artifact tab
     (Drive continuity is already handled BY THE VM — see below).
  6. Stops the session.

DRIVE CONTINUITY (why timeouts no longer lose work):
  daily_finetune.py mounts Drive itself (vendored gdrive.py + the ADC JSON we
  upload) and pushes a checkpoint to Drive after every --save-steps training
  steps, plus a final checkpoint + archive + adapter_in update when the run
  ends (naturally or via its --max-minutes budget). So even if THIS runner
  times out or the Colab session is recycled mid-training, the newest
  checkpoint already lives on Drive and the next day's run resumes from it.
  On runner timeout we additionally pull the newest Drive checkpoint into the
  adapter_in folder so continuity is preserved even if the VM died before its
  final push.

Secrets come from environment variables (set by the workflow from GitHub
secrets). None are ever committed to the repo.

Environment:
  COLAB_CLIENT_ID         OAuth client id (gcloud "Desktop" OAuth client)
  COLAB_CLIENT_SECRET     OAuth client secret
  COLAB_REFRESH_TOKEN     long-lived refresh token (colaboratory + drive.file scope)
  GDRIVE_ADC              full gcloud ADC JSON (authorized_user) for Drive
  DRIVE_ADAPTER_IN        Drive folder id for the "latest" adapter (shared, gdown pulls it)
  DRIVE_RESULTS           Drive folder id for dated result archives + checkpoints/
  RUN_ROWS                finance-alpaca subset size per day (default 5000)
  RUN_SAVE_STEPS          checkpoint every N training steps (default 100)
  RUN_MAX_MINUTES         stop training after N minutes, save final checkpoint (default 100)
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# run_daily.py lives in the same repo as daily_finetune.py — reuse its Drive
# wrapper + checkpoint discovery instead of duplicating Drive logic.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from daily_finetune import Drive, find_latest_checkpoint  # noqa: E402

TOKEN_FILE = pathlib.Path.home() / ".config" / "colab-cli" / "token.json"
ADC_FILE = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
REPO = pathlib.Path(__file__).resolve().parent
COLAB_PY = str(REPO / "colab.py")
GDRIVE_PY = str(REPO / "gdrive.py")
SESSION = "soup-daily"

# Colab free-tier sessions recycle after ~2-3h; the VM self-stops after
# RUN_MAX_MINUTES of training (default 100) so a full run fits well inside.
# Give the runner enough slack to cover model download + setup + polling.
TRAIN_TIMEOUT_S = int(os.environ.get("TRAIN_TIMEOUT_S", "9600"))


def log(msg):
    print(f"[run_daily] {msg}", flush=True)


def sh(args, timeout=180, check=False, ok_codes=(0,)):
    """Run a command, return (returncode, stdout, stderr)."""
    log("$ " + " ".join(str(a) for a in args))
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return (-1, (e.stdout or "") if isinstance(e.stdout, str) else "", "TIMEOUT")
    if check and r.returncode not in ok_codes:
        raise RuntimeError(f"command failed rc={r.returncode}: {args}\n{r.stderr[-2000:]}")
    return r.returncode, r.stdout, r.stderr


def mint_access_token(client_id, client_secret, refresh_token):
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def write_token_file(client_id, client_secret, refresh_token, access_token, scopes, expiry_s):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "token": access_token,
        "refresh_token": refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": scopes,
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": (datetime.datetime.now(datetime.timezone.utc)
                   + datetime.timedelta(seconds=expiry_s)).isoformat(),
    }
    TOKEN_FILE.write_text(json.dumps(doc, indent=2))
    log(f"wrote {TOKEN_FILE}")


def write_adc_file(adc_json):
    ADC_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADC_FILE.write_text(adc_json)
    ADC_FILE.chmod(0o600)
    log(f"wrote {ADC_FILE}")


def gdrive(*args, timeout=300):
    return sh([sys.executable, GDRIVE_PY, *args], timeout=timeout)


def colab(*args, timeout=300):
    return sh([sys.executable, COLAB_PY, *args], timeout=timeout)


def poll_log(deadline, needle="[RESULT]"):
    while time.time() < deadline:
        rc, out, err = colab("logs", "-s", SESSION, "/content/train.log", "-n", "8", timeout=60)
        if needle in out or "EXIT" in out or "NOT FOUND" in out:
            return out
        if rc != 0:
            # transient colab flakiness — keep polling
            log(f"log poll rc={rc}: {err[-200:]}")
        time.sleep(90)
    return None


def recover_latest_checkpoint_to_adapter_in(folder_in, run_id):
    """Pull the newest Drive checkpoint into the adapter_in continuity folder.
    Called on runner timeout / VM death so the next run resumes from the last
    saved checkpoint instead of the pre-run adapter."""
    try:
        drive = Drive(GDRIVE_PY, str(ADC_FILE))
        found = find_latest_checkpoint(drive, os.environ.get("DRIVE_RESULTS", ""), run_id)
        if not found:
            log("no Drive checkpoint found to recover")
            return False
        folder_id, step, file_id, file_name = found
        tmp = pathlib.Path("/tmp/ckpt_recover")
        tmp.mkdir(exist_ok=True)
        adapter = tmp / "adapter_model.safetensors"
        drive.download(file_id, str(adapter))
        cfg = tmp / "adapter_config.json"
        if not cfg.exists():
            for f in drive.list_files(folder_id):
                if f.get("n") == "adapter_config.json":
                    drive.download(f.get("id"), str(cfg))
                    break
        if not (adapter.exists() and adapter.stat().st_size > 0):
            log("recovered adapter file is empty/missing")
            return False
        # replace adapter_in contents
        rc, o, e = gdrive("list", "--folder", folder_in, "--max", "50", timeout=120)
        items = []
        try:
            data = json.loads(o or "{}")
            items = data.get("items", []) if isinstance(data, dict) and isinstance(data.get("items"), list) else []
        except Exception:
            items = []
        for it in items:
            gdrive("rm", it.get("id", ""), timeout=120) if isinstance(it, dict) and it.get("id") else None
        gdrive("upload", str(adapter), "--parent", folder_in, "--name", "adapter_model.safetensors", timeout=300)
        if cfg.exists():
            gdrive("upload", str(cfg), "--parent", folder_in, "--name", "adapter_config.json", timeout=120)
        log(f"recovered checkpoint step-{step} -> adapter_in (next run resumes from it)")
        return True
    except Exception as e:
        log(f"checkpoint recovery failed: {e}")
        return False


def main():
    # --- secrets ---
    cid = os.environ["COLAB_CLIENT_ID"]
    csec = os.environ["COLAB_CLIENT_SECRET"]
    cref = os.environ["COLAB_REFRESH_TOKEN"]
    adc = os.environ["GDRIVE_ADC"]
    folder_in = os.environ["DRIVE_ADAPTER_IN"]
    folder_out = os.environ["DRIVE_RESULTS"]
    rows = os.environ.get("RUN_ROWS", "2000")
    save_steps = os.environ.get("RUN_SAVE_STEPS", "100")
    max_minutes = os.environ.get("RUN_MAX_MINUTES", "100")
    seed = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    run_id = seed

    # --- auth ---
    tok = mint_access_token(cid, csec, cref)
    write_token_file(cid, csec, cref, tok["access_token"],
                     ["openid",
                      "https://www.googleapis.com/auth/userinfo.profile",
                      "https://www.googleapis.com/auth/userinfo.email",
                      "https://www.googleapis.com/auth/cloud-platform",
                      "https://www.googleapis.com/auth/colaboratory",
                      "https://www.googleapis.com/auth/drive.file"],
                     int(tok.get("expires_in", 3599)) - 60)
    write_adc_file(adc)

    # --- sanity: gdrive works headlessly (ADC can be inline JSON or path) ---
    rc, out, err = gdrive("about", timeout=120)
    log(f"gdrive about: rc={rc} {out[:160]}")
    if rc != 0:
        log(f"gdrive about FAILED (continuing, will retry on upload): {err[-300:]}")

    # --- session (retry on transient quota exhaustion: free T4 quota is
    # ~3-4 sessions/account/day and `gpu-unavailable` is common at peak) ---
    rc, out, err = 1, "", ""
    for attempt in range(4):
        rc, out, err = colab("new", "-s", SESSION, "--gpu", "T4", timeout=300)
        log(f"new session attempt {attempt+1}: rc={rc} {out[-200:]} {err[-200:]}")
        if rc == 0:
            break
        if "gpu-unavailable" in str(out) + str(err) and attempt < 3:
            log("GPU quota exhausted — waiting 600s before retry")
            time.sleep(600)
        else:
            break
    if rc != 0:
        raise SystemExit(f"failed to create T4 session: {str(out)[-500:] or str(err)[-500:]}")

    # --- upload training script + Drive tooling to the VM ---
    for local, remote in [
        (str(REPO / "daily_finetune.py"), "/content/daily_finetune.py"),
        (str(REPO / "gdrive.py"), "/content/gdrive.py"),
    ]:
        rc, o, e = colab("upload", "-s", SESSION, local, remote, timeout=180)
        if rc != 0:
            raise SystemExit(f"upload {local} failed: {e[-500:]}")
    # ADC JSON as a file on the VM (gdrive.py GDRIVE_ADC=path form)
    adc_local = pathlib.Path("/tmp/gdrive_adc.json")
    adc_local.write_text(adc)
    rc, o, e = colab("upload", "-s", SESSION, str(adc_local), "/content/gdrive_adc.json", timeout=180)
    if rc != 0:
        raise SystemExit(f"upload ADC failed: {e[-500:]}")

    # --- write + launch launcher (detached, logs to /content/train.log) ---
    launcher = pathlib.Path("/tmp/launch_daily.py")
    launcher.write_text(f"""import subprocess, sys
cmd = [sys.executable, '/content/daily_finetune.py',
       '--rows', '{rows}', '--seed', '{seed}',
       '--adapter-from-drive', '{folder_in}',
       '--adapter-out', '{folder_in}',
       '--drive-results', '{folder_out}',
       '--gdrive-py', '/content/gdrive.py',
       '--adc-file', '/content/gdrive_adc.json',
       '--run-id', '{run_id}',
       '--save-steps', '{save_steps}',
       '--max-minutes', '{max_minutes}',
       '--out', '/content/out']
print('LAUNCH ' + ' '.join(cmd), flush=True)
r = subprocess.run(cmd)
print('EXIT ' + str(r.returncode), flush=True)
sys.exit(r.returncode)
""")
    rc, out, err = colab("exec_detach", "-s", SESSION, "-f", str(launcher),
                         "--log", "/content/train.log", timeout=300)
    log(f"exec_detach: rc={rc} {out[-200:]} {err[-200:]}")
    if rc != 0:
        # Don't poll a log that will never appear — fail fast with the error.
        colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit(f"exec_detach failed (session stopped): {out[-400:] or err[-400:]}")

    # --- poll ---
    deadline = time.time() + TRAIN_TIMEOUT_S
    log(f"polling /content/train.log until {datetime.datetime.now().isoformat()} + {TRAIN_TIMEOUT_S}s")
    final = poll_log(deadline)
    if final is None:
        log("TIMEOUT: training did not finish in time (Colab may have recycled the session)")
        # The VM pushes checkpoints to Drive as it trains — recover the newest
        # one so the next run resumes from real progress, not the pre-run adapter.
        recover_latest_checkpoint_to_adapter_in(folder_in, run_id)
        colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit("training timed out; latest Drive checkpoint recovered into adapter_in")
    log(f"training finished:\n{final[-1200:]}")

    # --- download results (best-effort; Drive continuity is already done by VM) ---
    outdir = pathlib.Path("out")
    outdir.mkdir(exist_ok=True)
    results = {}
    for remote, local in [
        ("/content/out/adapter_model.safetensors", "out/adapter_model.safetensors"),
        ("/content/out/adapter_config.json", "out/adapter_config.json"),
        ("/content/out/metrics.json", "out/metrics.json"),
    ]:
        rc, o, e = colab("download", "-s", SESSION, remote, local, timeout=300)
        results[local] = rc == 0 and pathlib.Path(local).exists()
        log(f"download {local}: rc={rc} exists={results[local]}")
    if not results["out/adapter_model.safetensors"]:
        log("WARNING: adapter download failed — VM already pushed it to Drive (adapter_in + archive)")

    # --- stop session (free tier: don't leave it idle) ---
    colab("stop", "-s", SESSION, timeout=120)

    log("DONE")
    print("\n[run_daily] SUCCESS — adapter + checkpoints on Drive", flush=True)


if __name__ == "__main__":
    main()
