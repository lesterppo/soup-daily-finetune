#!/usr/bin/env python3
"""colab-cli v3.2 — AI-agent-native CLI for Google Colab.

Fixes over v3.0:
- P0: Retry logic for transient Colab errors (502/503/timeout), up to 2 retries
- P0: cmd_console shell injection fixed — json.dumps() escaping
- P0: tunnel_discover shell=True replaced with list-form subprocess
- P1: _remote_exists/_remote_read use json.dumps() for path safety
- P1: Auth pre-check window widened from 120s to 300s
- P1: shlex.split() for pip args (handles quoted arguments)
- P1: Browser sleeps replaced with wait_for_selector / wait_for_timeout
- P2: All =run_colab / =_remote_pexec spacing fixed

Wraps the official google-colab-cli with token-efficient output.
"""

import argparse, json, os, subprocess, sys, time, re, shlex, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
import threading

COLAB_HOME = Path.home() / ".hermes" / "scripts" / "colab"
COLAB_HOME.mkdir(parents=True, exist_ok=True)
TUNNEL_FILE = COLAB_HOME / "tunnel_url.json"
TOKEN_FILE = Path.home() / ".config" / "colab-cli" / "token.json"
CACHE_DIR = Path.home() / ".cache" / "colab-cli"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth token management ──────────────────────────────────────────────────

_token_lock = threading.Lock()
_token_stop_event = threading.Event()
_refresher_thread = None

def _find_colab():
    import shutil
    for name in ("colab", "google-colab-cli"):
        p = shutil.which(name)
        if p: return p
    for p in [os.path.expanduser("~/.local/bin/colab"), str(Path(sys.prefix)/"bin"/"colab")]:
        if os.path.isfile(p): return p
    return None

COLAB = _find_colab()

# The installed `colab` script may shebang to a broken python (e.g. a venv
# with mismatched pydantic_core). Prefer invoking the package via a working
# system python: `/usr/bin/python3 -m colab_cli.cli`.
COLAB_PY = None
for cand in ("/usr/bin/python3", "/usr/bin/python"):
    if os.path.exists(cand):
        COLAB_PY = cand
        break

# Patched shadow copy (created by `colabctl patch` when the installed package
# is root-owned): prefer it so the 401/404 -> refresh-token fix is active.
PATCHED_SHADOW = os.path.expanduser("~/colab_cli_patched")
if os.path.isdir(PATCHED_SHADOW):
    os.environ["PYTHONPATH"] = PATCHED_SHADOW + os.pathsep + os.environ.get("PYTHONPATH", "")

# The host shell (e.g. inside Hermes) may inject a venv's site-packages into
# PYTHONPATH, which breaks /usr/bin/python3's pydantic (mismatched
# pydantic_core). Strip venv/hermes paths from PYTHONPATH for every subprocess.
def _clean_env():
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    keep = [p for p in pp.split(os.pathsep)
            if p and "venv" not in p and ".hermes/hermes-agent" not in p]
    cleaned = os.pathsep.join(keep)
    env["PYTHONPATH"] = cleaned
    env.pop("VIRTUAL_ENV", None)
    return env


def _py_for_scripts():
    """System python for helper scripts (venv python has broken pydantic_core)."""
    return COLAB_PY or sys.executable

def die(code, msg):
    print(json.dumps({"ok": False, "err": code, "msg": msg}))
    sys.exit(1)

RETRYABLE_CODES = (502, 503, 504)  # Colab transient errors
MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds

def _is_retryable(stderr, rc):
    """Check if the error is transient and worth retrying."""
    combined = (stderr or "").lower()
    if rc in RETRYABLE_CODES:
        return True
    for phrase in ("connection reset", "connection refused", "service unavailable",
                   "temporarily unavailable", "backend error", "try again"):
        if phrase in combined:
            return True
    return False

def run_colab(args, timeout=120, stdin_data=None, env=None, retries=MAX_RETRIES):
    if COLAB_PY:
        cmd = [COLAB_PY, "-m", "colab_cli.cli"] + args
    else:
        cmd = [COLAB] + args
    merged_env = _clean_env()
    if env: merged_env.update(env)
    last_stdout, last_stderr, last_rc = "", "", -1
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env=merged_env, input=stdin_data)
            if r.returncode == 0 or not _is_retryable(r.stderr, r.returncode):
                return r.stdout, r.stderr, r.returncode
            last_stdout, last_stderr, last_rc = r.stdout, r.stderr, r.returncode
            # Wiped/expired session registry? Rebuild from the backend once,
            # then retry — this saved 40GB of model re-downloads in the field.
            combined = (r.stderr + r.stdout).lower()
            if attempt == 0 and ("session" in combined and "not found" in combined):
                _auto_recover()
            if attempt < retries:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except subprocess.TimeoutExpired:
            if attempt < retries and retries > 0:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return "", "timeout", -1
        except FileNotFoundError:
            return "", "colab binary not found", -2
    return last_stdout, last_stderr, last_rc

def _auto_recover():
    """Rebuild sessions.json from list_assignments() (same as cmd_recover)."""
    code = (
        "import sys; sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')\n"
        "from colab_cli.common import state\n"
        "import json, os\n"
        "try:\n"
        "    assignments = state.client.list_assignments()\n"
        "except Exception:\n"
        "    sys.exit(0)\n"
        "store_path = os.path.expanduser('~/.config/colab-cli/sessions.json')\n"
        "store = json.load(open(store_path)) if os.path.exists(store_path) else {}\n"
        "changed = False\n"
        "for a in assignments:\n"
        "    ep = a.endpoint\n"
        "    hit = None\n"
        "    for name, s in store.items():\n"
        "        if s.get('endpoint') == ep:\n"
        "            hit = (name, s); break\n"
        "    if hit:\n"
        "        name, s = hit\n"
        "        s['url'] = a.runtime_proxy_info.url\n"
        "        s['token'] = a.runtime_proxy_info.token\n"
        "        changed = True\n"
        "    else:\n"
        "        tail = ep.split('-')[-1][:10]\n"
        "        store['recovered_' + tail] = {\n"
        "            'name': ep, 'url': a.runtime_proxy_info.url,\n"
        "            'token': a.runtime_proxy_info.token, 'endpoint': ep,\n"
        "            'variant': getattr(a, 'variant', 'GPU'),\n"
        "            'accelerator': getattr(a, 'accelerator', 'T4')}\n"
        "        changed = True\n"
        "if changed:\n"
        "    json.dump(store, open(store_path, 'w'), indent=2)\n"
        "    print('AUTO_RECOVERED')\n"
    )
    try:
        tmp = COLAB_HOME / "auto_recover.py"
        tmp.write_text(code)
        r = subprocess.run([sys.executable, str(tmp)], capture_output=True,
                           text=True, timeout=60)
        if "AUTO_RECOVERED" in (r.stdout or ""):
            print("[colab] session registry was stale — rebuilt from backend", file=sys.stderr)
    except Exception:
        pass

def parse_error(stdout, stderr):
    combined = (stdout + "\n" + stderr).lower()
    if "auth" in combined and ("expired" in combined or "invalid" in combined):
        return "auth-expired", "Token expired."
    if "unauthorized" in combined or "authentication" in combined:
        return "auth-expired", "Auth required."
    if "rate" in combined and "limit" in combined:
        return "rate-limit", "Rate limited."
    if "quota" in combined:
        return "quota-exceeded", "Quota exceeded."
    if "not found" in combined or "does not exist" in combined:
        return "not-found", "Not found."
    if "timeout" in combined or "timed out" in combined:
        return "timeout", "Timed out."
    if "gpu" in combined and ("unavailable" in combined or "not available" in combined):
        return "gpu-unavailable", "GPU unavailable."
    if "busy" in combined or "capacity" in combined:
        return "capacity", "Backend busy."
    if "network" in combined or "connection" in combined:
        return "network", "Network error."
    return None, None

def write_output(text, out_file=None, json_mode=False, extra=None):
    if out_file:
        path = Path(out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        pointer = {"ok": True, "f": str(path.resolve()), "s": len(text.encode("utf-8")), "b": text.count("```") // 2}
        if extra: pointer.update(extra)
        print(json.dumps(pointer))
    elif json_mode:
        d = {"ok": True, "text": text}
        if extra: d.update(extra)
        print(json.dumps(d, ensure_ascii=False))
    else:
        print(text)

def check_rc(stdout, stderr, rc, fallback_code, fallback_msg):
    if rc != 0:
        err_code, err_msg = parse_error(stdout, stderr)
        if err_code:
            if err_code == "auth-expired":
                _try_refresh_token()
                # If refresh worked, caller can retry; but simplest is just to report
            die(err_code, err_msg or stderr.strip())
        die(fallback_code, stderr.strip() or stdout.strip() or fallback_msg)

def _try_refresh_token():
    """Attempt to refresh OAuth2 token. Returns True on success."""
    if not TOKEN_FILE.exists(): return False
    try:
        with _token_lock:
            creds = json.loads(TOKEN_FILE.read_text())
            if not creds.get("refresh_token"): return False
            data = urllib.parse.urlencode({
                "client_id": creds.get("client_id", ""),
                "client_secret": creds.get("client_secret", ""),
                "refresh_token": creds["refresh_token"],
                "grant_type": "refresh_token",
            }).encode()
            req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            resp = urllib.request.urlopen(req, timeout=15)
            t = json.loads(resp.read())
            creds["token"] = t["access_token"]
            creds["expiry"] = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + t.get("expires_in", 3600),
                tz=timezone.utc).isoformat()
            TOKEN_FILE.write_text(json.dumps(creds, indent=2))
            return True
    except Exception:
        return False

def _token_expires_soon(seconds_before=300):
    """Check if token expires within N seconds. Token expiry is stored as ISO string."""
    if not TOKEN_FILE.exists(): return False
    try:
        creds = json.loads(TOKEN_FILE.read_text())
        expiry_str = creds.get("expiry", "")
        if not expiry_str: return False
        # Handle both float epoch and ISO format
        if isinstance(expiry_str, (int, float)):
            expiry_ts = float(expiry_str)
        else:
            expiry_dt = datetime.fromisoformat(expiry_str.replace("+00:00", "").replace("Z", ""))
            expiry_ts = expiry_dt.timestamp()
        return (expiry_ts - datetime.now(timezone.utc).timestamp()) < seconds_before
    except Exception:
        return False

def _start_token_refresher():
    """Background thread: refresh token every 5 min if expiring soon."""
    global _refresher_thread
    if _refresher_thread and _refresher_thread.is_alive():
        return

    def _refresher_loop():
        while not _token_stop_event.is_set():
            try:
                if _token_expires_soon(300):
                    refreshed = _try_refresh_token()
                    if refreshed:
                        ts = datetime.now().strftime("%H:%M:%S")
                        sys.stderr.write(f"[colab-cli {ts}] Token auto-refreshed\n")
                        sys.stderr.flush()
            except Exception:
                pass
            _token_stop_event.wait(300)  # check every 5 min

    _refresher_thread = threading.Thread(target=_refresher_loop, daemon=True)
    _refresher_thread.start()

def _pre_auth_check():
    """Ensure auth is valid before command runs. Attempt refresh if needed."""
    if _token_expires_soon(300):
        _try_refresh_token()
    _start_token_refresher()

def format_sessions_list(stdout):
    sessions = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("─") or "Session" in line or "No active" in line: continue
        parts = line.split()
        if len(parts) >= 2: sessions.append({"name": parts[0], "status": parts[-1] if len(parts) > 2 else "active"})
    return sessions

def _resolve_session(args):
    """Auto-detect session if not specified."""
    if getattr(args, "session", None): return args.session
    stdout, stderr, rc = run_colab(["sessions"], timeout=15)
    if rc == 0:
        sessions = format_sessions_list(stdout)
        if len(sessions) == 1: return sessions[0]["name"]
    return None

# ─── Core VM helpers ──────────────────────────────────────────────────────

def _remote_exists(session, path):
    """Check if a file exists on the VM."""
    code = "import os; print('YES' if os.path.exists(%s) else 'NO')" % json.dumps(path)
    stdout, stderr, rc = run_colab(["exec", "-s", session, "--timeout", "5"], timeout=10, stdin_data=code)
    return rc == 0 and "YES" in stdout

def _remote_read(session, path):
    """Read a text file from the VM."""
    code = "print(open(%s).read())" % json.dumps(path)
    stdout, stderr, rc = run_colab(["exec", "-s", session, "--timeout", "10"], timeout=15, stdin_data=code)
    return stdout if rc == 0 else None

def _remote_pexec(session, code, timeout=10):
    """Run Python code on VM, return (stdout, stderr, rc)."""
    cargs = ["exec", "-s", session, "--timeout", str(max(timeout, 5))]
    return run_colab(cargs, timeout=timeout + 15, stdin_data=code)

# ─── Commands ──────────────────────────────────────────────────────────────

def cmd_new(args):
    _pre_auth_check()
    cargs = ["new"]
    if args.session: cargs.extend(["-s", args.session])
    if getattr(args, "gpu", None): cargs.extend(["--gpu", args.gpu])
    if getattr(args, "tpu", None): cargs.extend(["--tpu", args.tpu])
    stdout, stderr, rc = run_colab(cargs, timeout=120)
    check_rc(stdout, stderr, rc, "new-failed", "Failed to create session")
    name = args.session or stdout.strip().split()[-1]
    write_output(f"Session '{name}' created.\n{stdout.strip()}", args.out, args.json, {"session": name})

def cmd_exec(args):
    _pre_auth_check()
    cargs = ["exec"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "file", None): cargs.extend(["-f", args.file])
    if getattr(args, "output_image", None): cargs.extend(["--output-image", args.output_image])
    t = getattr(args, "timeout", 30) or 30
    cargs.extend(["--timeout", str(t)])
    stdin_data = args.code if getattr(args, "code", None) else None
    stdout, stderr, rc = run_colab(cargs, timeout=t + 60, stdin_data=stdin_data)
    check_rc(stdout, stderr, rc, "exec-failed", "Execution failed")
    write_output(stdout, args.out, args.json)

# ── P0: exec_detach — upload code + run detached, returns immediately ─────

def cmd_exec_detach(args):
    """Upload code file, run it with nohup/start_new_session, return PID immediately.

    This is THE way to launch long-running servers (llama.cpp, FastAPI, tunnels).
    Unlike exec (which blocks) or exec_bg (which runs Code as exec text),
    this uploads a script and runs it as a proper background process.
    """
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    if not s: die("exec-detach-no-session", "Specify -s SESSION or create one first")

    local_file = getattr(args, "file", None)
    code = getattr(args, "code", None)
    remote_path = getattr(args, "remote", None)
    log_file = getattr(args, "log", None) or "/dev/null"
    work_dir = getattr(args, "dir", None) or "/content"

    # Resolve local file path
    if local_file:
        local_path = Path(local_file)
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path
        if not local_path.exists():
            die("exec-detach-no-file", f"File not found: {local_file}")
        remote_path = remote_path or f"/content/{local_path.name}"

        # Upload the file
        stdout, stderr, rc = run_colab(["upload", "-s", s, str(local_path), remote_path], timeout=60)
        check_rc(stdout, stderr, rc, "upload-failed", "Upload failed")

    elif code:
        remote_path = remote_path or "/content/detach_script.py"
        # Write code to temp file and upload
        tmp = COLAB_HOME / "detach_code.py"
        tmp.write_text(code)
        stdout, stderr, rc = run_colab(["upload", "-s", s, str(tmp), remote_path], timeout=60)
        check_rc(stdout, stderr, rc, "upload-failed", "Upload failed")
    else:
        die("exec-detach-no-code", "Provide --file LOCAL or --code ...")

    # Launch detached on VM
    launch_code = (
        "import subprocess, sys, os\n"
        "p = subprocess.Popen(\n"
        "    [sys.executable, '%s'],\n"
        "    stdout=open('%s', 'w'), stderr=subprocess.STDOUT,\n"
        "    start_new_session=True, cwd='%s'\n"
        ")\n"
        "print(f'PID:{p.pid}|SESSION:%s|LOG:%s|SCRIPT:%s')"
    ) % (remote_path, log_file, work_dir, s, log_file, remote_path)

    stdout, stderr, rc = _remote_pexec(s, launch_code, timeout=15)
    if rc == 0 and "PID:" in stdout:
        info = {}
        for part in stdout.strip().split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                info[k] = v
        write_output("", args.out, args.json, {
            "status": "detached",
            "pid": info.get("PID"),
            "log": info.get("LOG"),
            "script": info.get("SCRIPT"),
            "session": s,
            "monitor": f"python3 ~/.hermes/scripts/colab/colab.py logs -s {s} {log_file.replace('/content/', '/content/')} -f"
        })
    else:
        die("exec-detach-failed", stdout.strip())


# ── P1: exec_file — upload + exec in one step ─────────────────────────────

def cmd_exec_file(args):
    """Upload a local script and execute it on the VM in one step.

    Combines upload + exec into a single command.
    """
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    if not s: die("exec-file-no-session", "Specify -s SESSION or create one first")

    local_path = Path(args.file)
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    if not local_path.exists():
        die("exec-file-not-found", f"File not found: {args.file}")

    remote_path = getattr(args, "remote", None) or f"/content/{local_path.name}"

    # Upload
    stdout, stderr, rc = run_colab(["upload", "-s", s, str(local_path), remote_path], timeout=60)
    check_rc(stdout, stderr, rc, "upload-failed", "Upload failed")

    # Execute
    t = getattr(args, "timeout", 30) or 30
    code = f"import subprocess, sys; r = subprocess.run([sys.executable, '{remote_path}'], capture_output=True, text=True, timeout={t}); print(r.stdout); print(r.stderr, file=sys.stderr if r.returncode else sys.stdout)"
    cargs = ["exec", "-s", s, "--timeout", str(t)]
    stdout, stderr, rc = run_colab(cargs, timeout=t + 60, stdin_data=code)
    check_rc(stdout, stderr, rc, "exec-file-failed", "Script execution failed")
    write_output(stdout, args.out, args.json, {"file": str(local_path), "remote": remote_path})


# ── P1: tunnel discover — auto-grep VM for live tunnel URL ────────────────

def cmd_tunnel_discover(args):
    """Search the VM for a live Cloudflare tunnel URL.

    Checks common locations: /content/tunnel.log, /content/api_url.txt,
    /kaggle/working/tunnel.log, /content/*.log patterns.
    """
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    if not s: die("tunnel-discover-no-session", "Specify -s SESSION or create one first")

    search_code = (
        "import os, re, glob\n"
        "urls = []\n"
        "patterns = [\n"
        "    '/content/tunnel.log', '/content/api_url.txt',\n"
        "    '/content/*.log', '/content/*deploy*.log',\n"
        "    '/content/server*.log', '/kaggle/working/tunnel.log',\n"
        "    '/kaggle/working/api_url.txt'\n"
        "]\n"
        "files_to_check = set()\n"
        "for pat in patterns:\n"
        "    files_to_check.update(glob.glob(pat))\n"
        "# Also check explicit paths\n"
        "for f in ['/content/tunnel.log', '/content/api_url.txt']:\n"
        "    if os.path.exists(f): files_to_check.add(f)\n"
        "for f in sorted(files_to_check):\n"
        "    try:\n"
        "        content = open(f).read()\n"
        "        found = re.findall(r'https://[a-zA-Z0-9.-]*\\.trycloudflare\\\\.com[^\\s\"\\'<>]*', content)\n"
        "        if found:\n"
        "            for u in found:\n"
        "                clean = u.rstrip('/')\n"
        "                if '/v1' not in clean:\n"
        "                    clean += '/v1'\n"
        "                urls.append({'url': clean, 'source': f})\n"
        "    except Exception:\n"
        "        pass\n"
        "# Also try running process check\n"
        "try:\n"
        "    import subprocess as _sp\n"
        "    r = _sp.run(['ps', 'aux'], capture_output=True, text=True)\n"
        "    if 'cloudflared' in r.stdout:\n"
        "        urls.append({'url': 'process_running', 'source': 'ps_aux'})\n"
        "except: pass\n"
        "if urls:\n"
        "    import json\n"
        "    print(json.dumps({'ok': True, 'tunnels': urls}))\n"
        "else:\n"
        "    print(json.dumps({'ok': False, 'err': 'not-found', 'msg': 'No tunnel URLs found on VM'}))\n"
    )
    stdout, stderr, rc = _remote_pexec(s, search_code, timeout=15)
    try:
        result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        result = {"ok": False, "err": "parse-error", "msg": stdout.strip()[:200]}

    if result.get("ok"):
        tunnels = result.get("tunnels", [])
        # Save all found URLs
        for t in tunnels:
            if "trycloudflare" in t.get("url", ""):
                data = json.loads(TUNNEL_FILE.read_text()) if TUNNEL_FILE.exists() else {}
                data[s] = t["url"]
                TUNNEL_FILE.write_text(json.dumps(data, indent=2))
        write_output(json.dumps(tunnels, indent=2), args.out, args.json, {"count": len(tunnels), "tunnels": tunnels})
    else:
        die("tunnel-discover-failed", result.get("msg", "No tunnels found"))


# ── P2: Fixed exec_bg with proper job tracking ────────────────────────────

def cmd_exec_bg(args):
    """Background execution — uploads code, runs it via exec_detach, returns job_id."""
    import uuid
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    if not s: die("exec-bg-no-session", "Specify -s SESSION or create one first")

    job_id = str(uuid.uuid4())[:8]
    timeout = getattr(args, "timeout", 600) or 600

    code = getattr(args, "code", None)
    if not code and not sys.stdin.isatty():
        code = sys.stdin.read().strip()
    if not code:
        die("exec-bg-no-code", "No code provided. Use --code or pipe via stdin.")

    # Write code to temp file
    code_file = COLAB_HOME / f"bg_{job_id}_code.py"
    code_file.write_text(code)
    remote_path = f"/tmp/bg_{job_id}.py"
    log_path = f"/tmp/bg_{job_id}.log"

    # Upload
    stdout, stderr, rc = run_colab(["upload", "-s", s, str(code_file), remote_path], timeout=60)
    check_rc(stdout, stderr, rc, "upload-failed", "Upload failed")

    # Launch detached
    launch_code = (
        "import subprocess, sys, os, json\n"
        "p = subprocess.Popen(\n"
        "    [sys.executable, '%s'],\n"
        "    stdout=open('%s', 'w'), stderr=subprocess.STDOUT,\n"
        "    start_new_session=True\n"
        ")\n"
        "print(json.dumps({'pid': p.pid, 'job_id': '%s', 'session': '%s'}))"
    ) % (remote_path, log_path, job_id, s)

    stdout, stderr, rc = _remote_pexec(s, launch_code, timeout=15)
    try:
        info = json.loads(stdout.strip())
    except json.JSONDecodeError:
        info = {"raw": stdout.strip()}

    write_output("", args.out, args.json, {
        "job_id": job_id,
        "pid": info.get("pid"),
        "status": "running",
        "session": s,
        "log": f"{{'code'}}logs -s {s} {log_path} -f",
        "poll": f"exec_bg_poll {job_id}",
        "code_file": str(code_file),
    })


def cmd_exec_bg_poll(args):
    """Poll a background execution job — checks VM log file."""
    _pre_auth_check()
    job_id = args.job_id
    session = getattr(args, "session", None)

    # Auto-detect session if not specified
    if not session:
        stdout, stderr, rc = run_colab(["sessions"], timeout=10)
        if rc == 0:
            slist = format_sessions_list(stdout)
            if slist:
                session = slist[0]["name"]

    if session:
        log_path = f"/tmp/bg_{job_id}.log"
        code = f"import os; print('EXISTS' if os.path.exists('{log_path}') else 'NO')"
        stdout, stderr, rc = _remote_pexec(session, code, timeout=5)
        if rc == 0 and "EXISTS" in stdout:
            content_text = _remote_read(session, log_path)
            if content_text:
                write_output(content_text, args.out, args.json, {
                    "job_id": job_id,
                    "session": session,
                    "status": "done",
                    "size": len(content_text)
                })
                return

    write_output("", args.out, args.json, {
        "job_id": job_id,
        "status": "running",
        "msg": "Job still running or session not found"
    })
def cmd_run(args):
    _pre_auth_check()
    cargs = ["run"]
    if getattr(args, "gpu", None): cargs.extend(["--gpu", args.gpu])
    if getattr(args, "tpu", None): cargs.extend(["--tpu", args.tpu])
    if getattr(args, "keep", None): cargs.append("--keep")
    if getattr(args, "file", None): cargs.append(args.file)
    if getattr(args, "args", None): cargs.extend(args.args)
    elif getattr(args, "code", None):
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".py"))
        tmp.write_text(args.code)
        cargs.append(str(tmp))
    stdout, stderr, rc = run_colab(cargs, timeout=300)
    check_rc(stdout, stderr, rc, "run-failed", "Job failed")
    write_output(stdout, args.out, args.json)


def cmd_sessions(args):
    _pre_auth_check()
    stdout, stderr, rc = run_colab(["sessions"], timeout=30)
    check_rc(stdout, stderr, rc, "sessions-failed", stdout.strip())
    sessions_data = format_sessions_list(stdout)
    if args.json: print(json.dumps({"ok": True, "sessions": sessions_data}))
    else: write_output(stdout, args.out, args.json, {"sessions": sessions_data})


def cmd_status(args):
    _pre_auth_check()
    cargs = ["status"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "status-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_stop(args):
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    cargs = ["stop"]
    if s: cargs.extend(["-s", s])
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "stop-failed", stdout.strip())
    if TUNNEL_FILE.exists():
        data = json.loads(TUNNEL_FILE.read_text())
        if s in data:
            del data[s]
            TUNNEL_FILE.write_text(json.dumps(data, indent=2))
    write_output(f"Session '{s}' stopped." if s else "Session stopped.", args.out, args.json)


def cmd_gpu_switch(args):
    s = _resolve_session(args) or args.session
    new_gpu = args.gpu
    if not new_gpu: die("gpu-switch-no-gpu", "Specify --gpu target")
    cargs = ["stop", "-s", s]
    run_colab(cargs, timeout=30)
    cargs = ["new", "-s", s, "--gpu", new_gpu]
    stdout3, stderr3, rc3 = run_colab(cargs, timeout=120)
    check_rc(stdout3, stderr3, rc3, "gpu-switch-failed", f"Failed to recreate with {new_gpu}")
    write_output(f"GPU switched to {new_gpu} on session '{s}'.\n{stdout3.strip()}", args.out, args.json, {"session": s, "gpu": new_gpu})


def cmd_ls(args):
    cargs = ["ls"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "path", None): cargs.append(args.path)
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "ls-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_upload(args):
    cargs = ["upload"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    cargs.extend([args.local, args.remote])
    stdout, stderr, rc = run_colab(cargs, timeout=60)
    check_rc(stdout, stderr, rc, "upload-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_download(args):
    cargs = ["download"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    cargs.extend([args.remote, args.local])
    stdout, stderr, rc = run_colab(cargs, timeout=60)
    check_rc(stdout, stderr, rc, "download-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_rm(args):
    cargs = ["rm"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    cargs.append(args.path)
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "rm-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_install(args):
    cargs = ["install"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "requirements", None): cargs.extend(["-r", args.requirements])
    pkgs = getattr(args, "packages", None)
    if pkgs: cargs.extend(pkgs)
    pip_args = getattr(args, "pip_args", None)
    if pip_args:
        code = f"import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', {json.dumps(' '.join(pkgs))}] + {json.dumps(shlex.split(pip_args))})\n"
        return cmd_exec_custom(args, s, code)
    stdout, stderr, rc = run_colab(cargs, timeout=120)
    check_rc(stdout, stderr, rc, "install-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_exec_custom(args, session, code):
    cargs = ["exec"]
    if session: cargs.extend(["-s", session])
    t = getattr(args, "timeout", 120) or 120
    cargs.extend(["--timeout", str(t)])
    stdout, stderr, rc = run_colab(cargs, timeout=t + 60, stdin_data=code)
    check_rc(stdout, stderr, rc, "exec-failed", "Execution failed")
    write_output(stdout, args.out, args.json)


def cmd_console(args):
    """Shell commands via exec+subprocess (avoids tmux garbling)."""
    s = _resolve_session(args) or args.session
    cmd_str = getattr(args, "cmd", None)
    if not cmd_str:
        if not sys.stdin.isatty():
            cmd_str = sys.stdin.read().strip()
    if not cmd_str:
        die("console-no-input", "No command provided. Use --command 'CMD' or pipe via stdin.")
    code = (
        "import subprocess\n"
        "r = subprocess.run(%s, shell=True, capture_output=True, text=True)\n"
        "if r.stdout: print(r.stdout, end='')\n"
        "if r.stderr: print(r.stderr, end='')"
    ) % json.dumps(cmd_str)
    cargs = ["exec"]
    if s: cargs.extend(["-s", s])
    t = getattr(args, "timeout", 60) or 60
    cargs.extend(["--timeout", str(t)])
    stdout, stderr, rc = run_colab(cargs, timeout=t + 30, stdin_data=code)
    check_rc(stdout, stderr, rc, "console-failed", stdout.strip() or stderr.strip())
    write_output(stdout, args.out, args.json)


def cmd_edit(args):
    cargs = ["edit"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    cargs.append(args.remote_path)
    stdout, stderr, rc = run_colab(cargs, timeout=60)
    check_rc(stdout, stderr, rc, "edit-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_repl(args):
    cargs = ["repl"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "output_image", None): cargs.extend(["--output-image", args.output_image])
    stdin_data = args.code if getattr(args, "code", None) else None
    stdout, stderr, rc = run_colab(cargs, timeout=getattr(args, "timeout", 60), stdin_data=stdin_data)
    check_rc(stdout, stderr, rc, "repl-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_log(args):
    cargs = ["log"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "lines", None): cargs.extend(["-n", str(args.lines)])
    if getattr(args, "event_type", None): cargs.extend(["-t", args.event_type])
    if getattr(args, "output", None): cargs.extend(["-o", args.output])
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "log-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_logs(args):
    """Tail a file on the Colab VM, with optional --follow streaming."""
    _pre_auth_check()
    s = _resolve_session(args) or args.session
    filepath = args.filepath
    lines = getattr(args, "lines", 50) or 50
    follow = getattr(args, "follow", False)

    if follow:
        import signal
        stop = [False]
        def on_sigint(sig, frame):
            stop[0] = True
        prev = signal.signal(signal.SIGINT, on_sigint)
        try:
            print(f"Streaming {filepath} (Ctrl+C to stop)...")
            last_size = 0
            while not stop[0]:
                code = (
                    "import os\ntry:\n"
                    "    size = os.path.getsize('%s')\n"
                    "    if size > %d:\n"
                    "        with open('%s') as f:\n"
                    "            f.seek(%d)\n"
                    "            print(f.read(), end='')\n"
                    "        print('\\n__SIZE__:' + str(size))\n"
                    "except Exception as e:\n"
                    "    print('__ERR__:' + str(e))"
                ) % (filepath, last_size, filepath, last_size)
                stdout, stderr, rc = _remote_pexec(s, code, timeout=10)
                if rc == 0 and stdout.strip():
                    if '__SIZE__:' in stdout:
                        content, marker = stdout.rsplit('__SIZE__:', 1)
                        if content.strip():
                            print(content, end='', flush=True)
                        try:
                            last_size = int(marker.strip().split('\n')[0])
                        except ValueError:
                            pass
                time.sleep(1.0)
        finally:
            signal.signal(signal.SIGINT, prev)
            if stop[0]:
                print("\nStopped.")
    else:
        code = (
            "import subprocess\n"
            "r = subprocess.run(['tail', '-n', str(%d), '%s'], capture_output=True, text=True)\n"
            "print(r.stdout, end='')\n"
            "if r.stderr: print(r.stderr, end='')"
        ) % (lines, filepath)
        stdout, stderr, rc = _remote_pexec(s, code, timeout=15)
        check_rc(stdout, "", rc, "logs-failed", stdout.strip())
        write_output(stdout, args.out, args.json)


def cmd_check(args):
    """Pre-flight model check: test imports before full deployment."""
    s = _resolve_session(args) or args.session
    code = args.code
    if not code:
        die("check-no-code", "Provide --code with Python to test")
    print(f"Running pre-flight check on {s or 'auto'}...")
    stdout, stderr, rc = _remote_pexec(s, code, timeout=getattr(args, "timeout", 60) or 60)
    if rc == 0:
        write_output(f"PASS:\n{stdout}", args.out, args.json, {"status": "pass"})
    else:
        write_output(f"FAIL:\n{stdout}", args.out, args.json, {"status": "fail"})


def cmd_notebook(args):
    s = _resolve_session(args) or args.session
    action = getattr(args, "nb_action", "export")
    if action == "export":
        out = getattr(args, "output", None) or f"/tmp/{s or 'colab'}_export.ipynb"
        cargs = ["log", "-o", out]
        if s: cargs.extend(["-s", s])
        stdout, stderr, rc = run_colab(cargs, timeout=30)
        check_rc(stdout, stderr, rc, "notebook-failed", stdout.strip())
        write_output(f"Notebook exported to: {out}\n{stdout.strip()}", args.out, args.json, {"notebook_path": out})
    elif action == "execute":
        nb_file = getattr(args, "file", None)
        if not nb_file: die("notebook-no-file", "Specify --file NOTEBOOK.ipynb")
        cargs = ["exec", "-f", nb_file]
        if s: cargs.extend(["-s", s])
        stdout, stderr, rc = run_colab(cargs, timeout=120)
        check_rc(stdout, stderr, rc, "notebook-exec-failed", stdout.strip())
        write_output(stdout, args.out, args.json)


def cmd_url(args):
    cargs = ["url"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "open_browser", None): cargs.append("--open")
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "url-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_drivemount(args):
    cargs = ["drivemount"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    if getattr(args, "path", None): cargs.append(args.path)
    stdout, stderr, rc = run_colab(cargs, timeout=60)
    check_rc(stdout, stderr, rc, "drivemount-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_auth(args):
    cargs = ["auth"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    stdout, stderr, rc = run_colab(cargs, timeout=60)
    check_rc(stdout, stderr, rc, "auth-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_restart(args):
    cargs = ["restart-kernel"]
    s = _resolve_session(args) or args.session
    if s: cargs.extend(["-s", s])
    stdout, stderr, rc = run_colab(cargs, timeout=30)
    check_rc(stdout, stderr, rc, "restart-failed", stdout.strip())
    write_output(stdout, args.out, args.json)


def cmd_pay(args):
    stdout, stderr, rc = run_colab(["pay"], timeout=10)
    check_rc(stdout, stderr, rc, "pay-failed", stdout.strip())
    write_output(stdout or "https://colab.research.google.com/signup", args.out, args.json)


def cmd_version(args):
    stdout, stderr, rc = run_colab(["version"], timeout=5)
    check_rc(stdout, stderr, rc, "version-failed", stdout.strip())
    write_output(f"colab-cli v3.2\n{stdout.strip()}", args.out, args.json)


def cmd_update(args):
    cargs = ["update"]
    if getattr(args, "install", None): cargs.append("--install")
    stdout, stderr, rc = run_colab(cargs, timeout=15)
    check_rc(stdout, stderr, rc, "update-failed", stdout.strip())
    write_output(stdout.strip(), args.out, args.json)


def cmd_whoami(args):
    stdout, stderr, rc = run_colab(["whoami"], timeout=10)
    if rc != 0:
        die("whoami-failed", "whoami not available. Upgrade: pip install --upgrade google-colab-cli")
    write_output(stdout.strip(), args.out, args.json)


# ─── Browser-based ─────────────────────────────────────────────────────────

def _get_browser_colab_url(session=None):
    cargs = ["url"]
    if session: cargs.extend(["-s", session])
    stdout, stderr, rc = run_colab(cargs, timeout=15)
    if rc != 0 or not stdout.strip(): return None
    for line in stdout.strip().splitlines():
        if line.strip().startswith("http"): return line.strip()
    return None


def cmd_secrets(args):
    action = getattr(args, "secrets_action", "list")
    key = getattr(args, "key", None)
    value = getattr(args, "value", None)
    s = _resolve_session(args) or args.session
    if action in ("set", "get", "delete") and not key:
        die("secrets-missing-key", f"Action '{action}' requires --key")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        die("secrets-no-playwright", "Install: pip install playwright && playwright install chromium")
    url = _get_browser_colab_url(s)
    if not url: die("secrets-no-url", "No browser URL. Session running?")
    result = {"ok": True, "action": action}
    pw = sync_playwright().start()
    browser = ctx = page = None
    try:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector('button:has-text("Secrets")', timeout=15000)
        secrets_btn = page.locator('button:has-text("Secrets")')
        if secrets_btn.count() == 0: die("secrets-not-found", "Secrets panel missing.")
        secrets_btn.first.click()
        page.wait_for_timeout(500)
        if action == "list":
            result["secrets"] = _parse_secrets_list(page.locator("body").inner_text())
        elif action == "set":
            add_btn = page.locator('button:has-text("Add new secret")')
            if add_btn.count() > 0:
                add_btn.first.click()
                page.wait_for_selector('input[placeholder*="Name"]', timeout=5000)
                ni = page.locator('input[placeholder*="Name"]')
                if ni.count() > 0: ni.first.fill(key)
                vi = page.locator('input[placeholder*="Value"]')
                if vi.count() > 0: vi.first.fill(value)
                page.keyboard.press("Enter")
                page.wait_for_timeout(500)
                result["msg"] = f"Secret '{key}' set."
        elif action == "delete":
            result["msg"] = "Delete requires DOM interaction — use list first to identify."
    except Exception as e:
        die("secrets-error", str(e))
    finally:
        for obj in [page, ctx, browser]:
            try: obj.close()
            except: pass
        try: pw.stop()
        except: pass
    write_output(json.dumps(result, ensure_ascii=False), args.out, args.json)


def _parse_secrets_list(page_text):
    secrets = []
    in_secrets = False
    for line in page_text.split("\n"):
        s = line.strip()
        if "Secrets" in s and "Close" in s: in_secrets = True; continue
        if in_secrets and s == "Close": break
        if in_secrets and s and not s.startswith("Add") and not s.startswith("Secrets"): secrets.append(s)
    return secrets


def cmd_resources(args):
    s = _resolve_session(args) or args.session
    try:
        from playwright.sync_api import sync_playwright
        url = _get_browser_colab_url(s)
        if url:
            pw = sync_playwright().start()
            browser = ctx = page = None
            try:
                browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
                ctx = browser.new_context(viewport={"width": 1280, "height": 800})
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1000)
                body = page.locator("body").inner_text()
                resources = {}
                for m in re.finditer(r'RAM:\s*([\d.]+)\s*GB\s*/\s*([\d.]+)\s*GB', body):
                    u, t = float(m.group(1)), float(m.group(2))
                    resources["ram_used_gb"] = u; resources["ram_total_gb"] = t
                    resources["ram_percent"] = round(u / t * 100, 1) if t else 0
                for m in re.finditer(r'Disk:\s*([\d.]+)\s*GB\s*/\s*([\d.]+)\s*GB', body):
                    u, t = float(m.group(1)), float(m.group(2))
                    resources["disk_used_gb"] = u; resources["disk_total_gb"] = t
                    resources["disk_percent"] = round(u / t * 100, 1) if t else 0
                if resources:
                    write_output(json.dumps({"ok": True, "resources": resources}, ensure_ascii=False), args.out, args.json)
                    return
            finally:
                for obj in [page, ctx, browser]:
                    try: obj.close()
                    except: pass
                try: pw.stop()
                except: pass
    except Exception: pass
    cmd_status(args)


def cmd_share(args):
    s = _resolve_session(args) or args.session
    url = _get_browser_colab_url(s)
    if not url: die("share-no-url", "No browser URL.")
    result = {"ok": True, "url": url, "msg": "Share this URL."}
    write_output(json.dumps(result, ensure_ascii=False), args.out, args.json)


def cmd_tunnel(args):
    """Get or save Cloudflare tunnel URL for a session."""
    s = _resolve_session(args) or args.session
    action = getattr(args, "action", "get")
    if action == "get":
        if TUNNEL_FILE.exists():
            data = json.loads(TUNNEL_FILE.read_text())
            url = data.get(s) or data.get("default")
            if url:
                write_output(url, args.out, args.json, {"tunnel_url": url, "session": s})
                return
        die("tunnel-not-found", "No tunnel URL saved. Run: colab.py tunnel set --url URL -s SESSION")
    elif action == "set":
        url = getattr(args, "url", None)
        if not url: die("tunnel-no-url", "Provide --url")
        data = {}
        if TUNNEL_FILE.exists(): data = json.loads(TUNNEL_FILE.read_text())
        data[s or "default"] = url
        TUNNEL_FILE.write_text(json.dumps(data, indent=2))
        write_output(f"Tunnel URL saved for {s or 'default'}: {url}", args.out, args.json)


# ─── P1: recover — rebuild sessions.json from the backend ──────────────────
# The colab-cli registry (sessions.json) can be wiped by crashes or the
# upstream CLI's destructive cleanup. This rebuilds it from the server's
# list_assignments() so a live VM is NOT lost (a wiped registry previously
# caused 40GB of models to be re-downloaded).

def cmd_recover(args):
    code = (
        "import sys; sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')\n"
        "from colab_cli.common import state\n"
        "import json, os\n"
        "try:\n"
        "    assignments = state.client.list_assignments()\n"
        "except Exception as e:\n"
        "    print('LIST_FAIL ' + str(e)[:200]); sys.exit(1)\n"
        "store_path = os.path.expanduser('~/.config/colab-cli/sessions.json')\n"
        "store = json.load(open(store_path)) if os.path.exists(store_path) else {}\n"
        "changed = False\n"
        "for a in assignments:\n"
        "    ep = a.endpoint\n"
        "    hit = None\n"
        "    for name, s in store.items():\n"
        "        if s.get('endpoint') == ep:\n"
        "            hit = (name, s); break\n"
        "    if hit:\n"
        "        name, s = hit\n"
        "        s['url'] = a.runtime_proxy_info.url\n"
        "        s['token'] = a.runtime_proxy_info.token\n"
        "        changed = True\n"
        "    else:\n"
        "        tail = ep.split('-')[-1][:10]\n"
        "        store['recovered_' + tail] = {\n"
        "            'name': ep, 'url': a.runtime_proxy_info.url,\n"
        "            'token': a.runtime_proxy_info.token, 'endpoint': ep,\n"
        "            'variant': getattr(a, 'variant', 'GPU'),\n"
        "            'accelerator': getattr(a, 'accelerator', 'T4')}\n"
        "        changed = True\n"
        "if changed:\n"
        "    json.dump(store, open(store_path, 'w'), indent=2)\n"
        "    print('RECOVERED ' + json.dumps({k: v.get('endpoint') for k, v in store.items()}))\n"
        "else:\n"
        "    print('NO_ASSIGNMENTS')\n"
    )
    tmp = COLAB_HOME / "recover.py"
    tmp.write_text(code)
    try:
        r = subprocess.run([_py_for_scripts(), str(tmp)], capture_output=True,
                           text=True, timeout=120, env=_clean_env())
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        out = "timeout"
    if "RECOVERED" in out:
        write_output(out, args.out, args.json, {"status": "recovered"})
    elif "NO_ASSIGNMENTS" in out:
        write_output("No live assignments found. Is the VM still running?",
                     args.out, args.json, {"status": "no-assignments"})
    else:
        die("recover-failed", out.strip()[-400:])


# ─── P2: reauth — switch the Colab account (manual OAuth flow) ─────────────
# Free-tier GPU quota is per-account; when `new` returns 503 outcome:2 the
# account is exhausted for the day. Re-auth as another Google account and
# keep the same session name. The flow prints an AUTH_URL; open it in a
# browser, approve, and the token is saved.

def cmd_reauth(args):
    hint = getattr(args, "account", None)
    code = f'''
import sys, os, json, threading, time
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
os.environ.setdefault("PYTHONPATH", "")
from colab_cli.auth import PUBLIC_SCOPES
from google_auth_oauthlib.flow import InstalledAppFlow
PORT = 8765
cfg = None
for c in [os.path.expanduser("~/.colab-cli-oauth-config.json"),
          "/usr/local/lib/python3.12/dist-packages/colab_cli/oauth_config.json"]:
    if os.path.exists(c):
        cfg = json.load(open(c)); break
if not cfg:
    print("NO_CLIENT_CONFIG"); sys.exit(1)
flow = InstalledAppFlow.from_client_config(cfg, PUBLIC_SCOPES)
flow.redirect_uri = f"http://localhost:{{PORT}}"
kwargs = dict(access_type="offline", prompt="consent",
              include_granted_scopes="true")
{"if hint: kwargs['login_hint'] = %r" % hint}
auth_url, _ = flow.authorization_url(**kwargs)
print("AUTH_URL: " + auth_url, flush=True)
holder = {{}}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        if "code" in q:
            holder["code"] = q["code"][0]
            self.send_response(200); self.end_headers()
            self.wfile.write(b"OK - close this tab")
        else:
            self.send_response(400); self.end_headers(); self.wfile.write(b"no code")
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
deadline = time.time() + 900
while "code" not in holder and time.time() < deadline:
    time.sleep(1)
if "code" not in holder:
    print("TIMEOUT"); sys.exit(1)
flow.fetch_token(code=holder["code"])
open(os.path.expanduser("~/.config/colab-cli/token.json"), "w").write(
    flow.credentials.to_json())
print("SAVED " + ",".join(flow.credentials.scopes or []), flush=True)
'''
    tmp = COLAB_HOME / "reauth_flow.py"
    tmp.write_text(code)
    print("Open the AUTH_URL below in a browser, approve, and the CLI account "
          "switches. (See references/auth_flow.md for the manual flow.)")
    r = subprocess.run([_py_for_scripts(), str(tmp)], capture_output=True,
                       text=True, timeout=960, env=_clean_env())
    out = (r.stdout or "") + (r.stderr or "")
    if "SAVED" in out:
        write_output("Token saved; scopes: " + out.split("SAVED ", 1)[1],
                     args.out, args.json, {"status": "reauth-ok"})
    else:
        die("reauth-failed", out.strip()[-400:])


# ─── P3: patch — apply the 401/404-refresh reliability patch ──────────────
def cmd_patch(args):
    here = Path(__file__).resolve().parent
    script = here / "patch_colab_cli.py"
    if not script.exists():
        die("patch-no-script", f"{script} not found in repo")
    r = subprocess.run([_py_for_scripts(), str(script)], capture_output=True,
                       text=True, timeout=120, env=_clean_env())
    out = (r.stdout or "") + (r.stderr or "")
    if "patched" in out or "already patched" in out:
        write_output(out.strip(), args.out, args.json, {"status": "patched"})
    else:
        die("patch-failed", out.strip()[-400:])


# ─── CLI ───────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="colab-cli v3.2 — AI-agent-native CLI for Google Colab",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    shared_output = argparse.ArgumentParser(add_help=False)
    shared_output.add_argument("-o", "--out", help="Write output to FILE, stdout gets JSON pointer")
    shared_output.add_argument("--json", action="store_true", help="Structured JSON output")

    sub = p.add_subparsers(dest="command")

    # Session management
    pn = sub.add_parser("new", parents=[shared_output], help="Create session")
    pn.add_argument("-s", "--session"); pn.add_argument("--gpu", choices=["T4","L4","G4","H100","A100"]); pn.add_argument("--tpu", choices=["v5e1","v6e1"])
    sub.add_parser("sessions", parents=[shared_output], help="List sessions")
    ps = sub.add_parser("status", parents=[shared_output], help="Session status"); ps.add_argument("-s", "--session")
    ps2 = sub.add_parser("stop", parents=[shared_output], help="Stop session"); ps2.add_argument("-s", "--session")
    pr = sub.add_parser("restart", parents=[shared_output], help="Restart kernel"); pr.add_argument("-s", "--session")
    pg = sub.add_parser("gpu_switch", parents=[shared_output], help="Switch GPU on session"); pg.add_argument("-s", "--session"); pg.add_argument("--gpu", choices=["T4","L4","G4","H100","A100"], required=True)

    # Execution
    pe = sub.add_parser("exec", parents=[shared_output], help="Execute code on VM")
    pe.add_argument("-s", "--session"); pe.add_argument("-f", "--file"); pe.add_argument("--code"); pe.add_argument("--output-image"); pe.add_argument("--timeout", type=float, default=30.0)

    # NEW: exec_detach — upload + run server code detached
    ped = sub.add_parser("exec_detach", parents=[shared_output], help="Upload script + run detached (for servers)")
    ped.add_argument("-s", "--session", required=True, help="Colab session name")
    ped.add_argument("-f", "--file", help="Local Python script to upload and run")
    ped.add_argument("--code", help="Python code to upload and run (if no -f)")
    ped.add_argument("--remote", help="Remote path (default: /content/<filename>)")
    ped.add_argument("--log", help="Log file path on VM (default: /dev/null)")
    ped.add_argument("--dir", default="/content", help="Working directory on VM (default: /content)")

    # NEW: exec_file — upload + exec in one step
    pef = sub.add_parser("exec_file", parents=[shared_output], help="Upload and execute a script in one step")
    pef.add_argument("-s", "--session", required=True, help="Colab session name")
    pef.add_argument("-f", "--file", required=True, help="Local Python script")
    pef.add_argument("--remote", help="Remote path (default: /content/<filename>)")
    pef.add_argument("--timeout", type=float, default=120.0, help="Script timeout in seconds")

    # Fixed exec_bg
    peb = sub.add_parser("exec_bg", parents=[shared_output], help="Background execution on VM")
    peb.add_argument("-s", "--session"); peb.add_argument("--code"); peb.add_argument("--timeout", type=float, default=600.0)
    pep = sub.add_parser("exec_bg_poll", parents=[shared_output], help="Poll background job"); pep.add_argument("job_id"); pep.add_argument("-s", "--session")

    # Other execution
    pr2 = sub.add_parser("run", parents=[shared_output], help="Ephemeral job"); pr2.add_argument("-f", "--file"); pr2.add_argument("--code"); pr2.add_argument("--gpu", choices=["T4","L4","G4","H100","A100"]); pr2.add_argument("--tpu", choices=["v5e1","v6e1"]); pr2.add_argument("--keep", action="store_true"); pr2.add_argument("args", nargs="*")
    pr3 = sub.add_parser("repl", parents=[shared_output], help="Python REPL"); pr3.add_argument("-s", "--session"); pr3.add_argument("--code"); pr3.add_argument("--output-image"); pr3.add_argument("--timeout", type=float, default=60.0)
    pc = sub.add_parser("console", parents=[shared_output], help="Shell command on VM"); pc.add_argument("-s", "--session"); pc.add_argument("--cmd", help="Shell command to run"); pc.add_argument("--timeout", type=float, default=60.0)

    # NEW: tunnel discover
    ptd = sub.add_parser("tunnel_discover", parents=[shared_output], help="Auto-discover live tunnel URL from VM")
    ptd.add_argument("-s", "--session", required=True, help="Colab session name")

    # VM file ops
    plogs = sub.add_parser("logs", parents=[shared_output], help="Tail VM file")
    plogs.add_argument("-s", "--session"); plogs.add_argument("filepath", help="Path on VM (e.g. /content/server.log)")
    plogs.add_argument("-n", "--lines", type=int, default=50, help="Lines to show")
    plogs.add_argument("-f", "--follow", action="store_true", help="Stream output (Ctrl+C to stop)")

    # File ops
    pl = sub.add_parser("ls", parents=[shared_output], help="List files on VM"); pl.add_argument("-s", "--session"); pl.add_argument("path", nargs="?")
    pu = sub.add_parser("upload", parents=[shared_output], help="Upload file to VM"); pu.add_argument("-s", "--session"); pu.add_argument("local"); pu.add_argument("remote")
    pd = sub.add_parser("download", parents=[shared_output], help="Download file from VM"); pd.add_argument("-s", "--session"); pd.add_argument("remote"); pd.add_argument("local")
    prm = sub.add_parser("rm", parents=[shared_output], help="Delete file on VM"); prm.add_argument("-s", "--session"); prm.add_argument("path")
    pe2 = sub.add_parser("edit", parents=[shared_output], help="Edit remote file"); pe2.add_argument("-s", "--session"); pe2.add_argument("remote_path")

    # Automation
    pi = sub.add_parser("install", parents=[shared_output], help="Install packages on VM"); pi.add_argument("-s", "--session"); pi.add_argument("-r", "--requirements"); pi.add_argument("--pip-args", help="Extra pip flags"); pi.add_argument("packages", nargs="*")
    pchk = sub.add_parser("check", parents=[shared_output], help="Pre-flight model test"); pchk.add_argument("-s", "--session"); pchk.add_argument("--code", required=True, help="Python to test"); pchk.add_argument("--timeout", type=float, default=60.0)
    pl2 = sub.add_parser("log", parents=[shared_output], help="View/export history"); pl2.add_argument("-s", "--session"); pl2.add_argument("-n", "--lines", type=int); pl2.add_argument("-t", "--event-type"); pl2.add_argument("--output")
    pn2 = sub.add_parser("notebook", parents=[shared_output], help="Notebook export/execute"); pn2.add_argument("-s", "--session"); pn2.add_argument("action", nargs="?", choices=["export","execute"], default="export", help="export or execute"); pn2.add_argument("--output", help="Export path"); pn2.add_argument("--file", help="Notebook file to execute")
    purl = sub.add_parser("url", parents=[shared_output], help="Get browser URL"); purl.add_argument("-s", "--session"); purl.add_argument("--open", dest="open_browser", action="store_true")
    pdm = sub.add_parser("drivemount", parents=[shared_output], help="Mount Drive"); pdm.add_argument("-s", "--session"); pdm.add_argument("path", nargs="?")
    pa = sub.add_parser("auth", parents=[shared_output], help="Auth VM for GCP"); pa.add_argument("-s", "--session")

    # Info
    sub.add_parser("pay", parents=[shared_output], help="Subscription page")
    sub.add_parser("version", parents=[shared_output], help="CLI version")
    pu2 = sub.add_parser("update", parents=[shared_output], help="Check updates"); pu2.add_argument("--install", action="store_true")
    sub.add_parser("whoami", parents=[shared_output], help="Auth identity")

    # Browser
    ps3 = sub.add_parser("secrets", parents=[shared_output], help="Manage secrets"); ps3.add_argument("-s", "--session"); ps3.add_argument("action", nargs="?", default="list", choices=["list","set","get","delete"]); ps3.add_argument("--key"); ps3.add_argument("--value")
    pr4 = sub.add_parser("resources", parents=[shared_output], help="VM resources"); pr4.add_argument("-s", "--session")
    psh = sub.add_parser("share", parents=[shared_output], help="Share URL"); psh.add_argument("-s", "--session")
    pt = sub.add_parser("tunnel", parents=[shared_output], help="Tunnel URL mgmt"); pt.add_argument("-s", "--session"); pt.add_argument("action", nargs="?", default="get", choices=["get","set"]); pt.add_argument("--url")
    prc = sub.add_parser("recover", parents=[shared_output], help="Rebuild sessions.json from backend (fixes wiped registry)")
    pra = sub.add_parser("reauth", parents=[shared_output], help="Re-auth / switch Colab account (per-account GPU quota)"); pra.add_argument("--account", help="login_hint email")
    ppp = sub.add_parser("patch", parents=[shared_output], help="Patch installed colab_cli: 401/404 -> refresh token + retry")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command: parser.print_help(); sys.exit(1)
    if not COLAB or not os.path.isfile(COLAB):
        die("no-colab-cli", "google-colab-cli not found. Install: pip install google-colab-cli")

    dispatcher = {
        "new": cmd_new, "exec": cmd_exec, "exec_bg": cmd_exec_bg, "exec_bg_poll": cmd_exec_bg_poll,
        "exec_detach": cmd_exec_detach, "exec_file": cmd_exec_file,
        "run": cmd_run, "sessions": cmd_sessions, "status": cmd_status, "stop": cmd_stop,
        "gpu_switch": cmd_gpu_switch, "restart": cmd_restart,
        "ls": cmd_ls, "upload": cmd_upload, "download": cmd_download, "rm": cmd_rm,
        "install": cmd_install, "log": cmd_log, "notebook": cmd_notebook,
        "url": cmd_url, "drivemount": cmd_drivemount, "auth": cmd_auth,
        "console": cmd_console, "edit": cmd_edit, "repl": cmd_repl,
        "logs": cmd_logs, "check": cmd_check,
        "tunnel": cmd_tunnel, "tunnel_discover": cmd_tunnel_discover,
        "pay": cmd_pay, "version": cmd_version, "update": cmd_update, "whoami": cmd_whoami,
        "secrets": cmd_secrets, "resources": cmd_resources, "share": cmd_share,
        "recover": cmd_recover, "reauth": cmd_reauth, "patch": cmd_patch,
    }
    try:
        dispatcher[args.command](args)
    except KeyboardInterrupt: die("interrupted", "Cancelled")
    except Exception as e: die("exception", str(e))


if __name__ == "__main__":
    main()
