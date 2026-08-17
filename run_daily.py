#!/usr/bin/env python3
"""
run_daily.py — headless daily LoRA fine-tune orchestrator.

Runs inside GitHub Actions (no local machine involved). It:
  1. Mints a Colab access token from a stored refresh token (server-side OAuth,
     no browser).
  2. Writes the colab-cli token.json and the gdrive ADC credentials.
  3. Creates a free Colab T4 session, uploads daily_finetune.py, and launches
     training detached.
  4. Polls the VM log until the run reports [RESULT] (or times out).
  5. Downloads the new adapter + metrics, uploads them to Google Drive
     (a dated archive folder AND the "latest" adapter_in folder the next day
     continues from).
  6. Stops the session.

Secrets come from environment variables (set by the workflow from GitHub
secrets). None are ever committed to the repo.

Environment:
  COLAB_CLIENT_ID         OAuth client id (gcloud client 764086051850-*)
  COLAB_CLIENT_SECRET     OAuth client secret
  COLAB_REFRESH_TOKEN     long-lived refresh token (colaboratory + drive.file scope)
  GDRIVE_ADC              full gcloud ADC JSON (authorized_user) for Drive
  DRIVE_ADAPTER_IN        Drive folder id for the "latest" adapter (shared, gdown pulls it)
  DRIVE_RESULTS           Drive folder id for dated result archives
  RUN_ROWS                finance-alpaca subset size per day (default 5000)
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

TOKEN_FILE = pathlib.Path.home() / ".config" / "colab-cli" / "token.json"
ADC_FILE = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
REPO = pathlib.Path(__file__).resolve().parent
COLAB_PY = str(REPO / "colab.py")
GDRIVE_PY = str(REPO / "gdrive.py")
SESSION = "soup-daily"

# Colab free-tier session lifetime is ~2-3h; give training 100 min before we
# declare failure (base-model download is ~8 min, then ~90 min of training).
TRAIN_TIMEOUT_S = int(os.environ.get("TRAIN_TIMEOUT_S", "6600"))


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


def main():
    # --- secrets ---
    cid = os.environ["COLAB_CLIENT_ID"]
    csec = os.environ["COLAB_CLIENT_SECRET"]
    cref = os.environ["COLAB_REFRESH_TOKEN"]
    adc = os.environ["GDRIVE_ADC"]
    folder_in = os.environ["DRIVE_ADAPTER_IN"]
    folder_out = os.environ["DRIVE_RESULTS"]
    rows = os.environ.get("RUN_ROWS", "2000")
    seed = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

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

    # --- sanity: gdrive works headlessly ---
    rc, out, err = gdrive("about", timeout=120)
    log(f"gdrive about: rc={rc} {out[:120]}")
    if rc != 0:
        log(f"gdrive about FAILED: {err[-500:]}")

    # --- session ---
    rc, out, err = colab("new", "-s", SESSION, "--gpu", "T4", timeout=300)
    log(f"new session: rc={rc} {out[-200:]} {err[-200:]}")
    if rc != 0:
        raise SystemExit(f"failed to create T4 session: {err[-500:]}")

    # --- upload training script ---
    rc, out, err = colab("upload", "-s", SESSION,
                         str(REPO / "daily_finetune.py"), "/content/daily_finetune.py",
                         timeout=180)
    if rc != 0:
        raise SystemExit(f"upload failed: {err[-500:]}")

    # --- write + launch launcher (detached, logs to /content/train.log) ---
    launcher = pathlib.Path("/tmp/launch_daily.py")
    launcher.write_text(f"""import subprocess, sys
cmd = [sys.executable, '/content/daily_finetune.py',
       '--rows', '{rows}', '--seed', '{seed}',
       '--adapter-from-drive', '{folder_in}',
       '--out', '/content/out', '--skip-install']
print('LAUNCH ' + ' '.join(cmd), flush=True)
r = subprocess.run(cmd)
print('EXIT ' + str(r.returncode), flush=True)
sys.exit(r.returncode)
""")
    rc, out, err = colab("exec_detach", "-s", SESSION, "-f", str(launcher),
                         "--log", "/content/train.log", timeout=180)
    log(f"exec_detach: rc={rc} {out[-120:]} {err[-120:]}")

    # --- poll ---
    deadline = time.time() + TRAIN_TIMEOUT_S
    log(f"polling /content/train.log until {datetime.datetime.now().isoformat()} + {TRAIN_TIMEOUT_S}s")
    final = poll_log(deadline)
    if final is None:
        log("TIMEOUT: training did not finish in time")
        colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit("training timed out (session likely recycled)")
    log(f"training finished:\n{final[-1200:]}")

    # --- download results ---
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
        colab("stop", "-s", SESSION, timeout=120)
        raise SystemExit("adapter download failed")

    # --- stop session (free tier: don't leave it idle) ---
    colab("stop", "-s", SESSION, timeout=120)

    # --- upload to Drive ---
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    # archive under results/<date>/
    rc, o, e = gdrive("mkdir", f"results-{date}", "--parent", folder_out, timeout=120)
    archive = None
    try:
        archive = json.loads(o).get("id")
    except Exception:
        log(f"mkdir archive failed: {o[:200]} {e[:200]}")
    if archive:
        for name in ["adapter_model.safetensors", "adapter_config.json", "metrics.json"]:
            if pathlib.Path("out", name).exists():
                gdrive("upload", f"out/{name}", "--parent", archive, "--name", name, timeout=300)

    # update the "latest" adapter_in folder: trash existing files, upload new ones.
    # (gdown --folder pulls whatever non-trashed files remain, so the next day's
    #  run continues from today's adapter.)
    rc, o, e = gdrive("list", "--folder", folder_in, "--max", "50", timeout=120)
    try:
        items = json.loads(o).get("items", [])
    except Exception:
        items = []
    for it in items:
        gdrive("rm", it["id"], timeout=120)  # trash (recoverable), not permanent
    for name in ["adapter_model.safetensors", "adapter_config.json"]:
        gdrive("upload", f"out/{name}", "--parent", folder_in, "--name", name, timeout=300)

    log("DONE")
    print("\n[run_daily] SUCCESS — adapter updated on Drive", flush=True)


if __name__ == "__main__":
    main()
