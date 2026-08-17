#!/usr/bin/env python3
"""gdrive — AI-agent-native token-efficient Google Drive CLI.

Full-operation Google Drive via the Drive v3 REST API. Compact JSON on
stdout, full payloads on disk, structured errors. No MCP, no browser.

Output contract (agent-native):
  - Success: compact JSON, short keys. Large payloads (list/tree/search)
    are written to a file and stdout carries a pointer {f, n}.
  - Error: {"e": "<msg>"} on stdout, non-zero exit.
  - Human text (progress/status) goes to stderr, never stdout.

Auth:
  - OAuth2 user token: ~/.hermes/google_token.json
    (shared with the google-workspace skill; auto-refreshes)
  - Client secret: ~/.hermes/google_client_secret.json
    (Google Cloud Console -> Credentials -> OAuth 2.0 Client ID, Desktop app)
  - Overrides: GDRIVE_TOKEN, GDRIVE_SECRET env vars
  - `gdrive auth` runs the PKCE setup flow (drive scope only).

Actions:
  auth            OAuth2 setup / status / revoke
  about           account + storage quota
  list            list files (folder / query / mime / trashed filters)
  search          full-text search
  get             metadata for one file or folder
  find            resolve a path like "a/b/c" to an id (walk by name)
  tree            recursive folder listing (depth-limited)
  upload          upload a local file (simple or resumable)
  import          upload + convert to a Google format (docx->doc, csv->sheets, ...)
  download        download a binary file to disk
  export          export a Google-native file (doc/sheet/slide/draw) to a format
  mkdir           create a folder
  touch           create an empty Google file (doc/sheet/slide)
  mv              move file(s) into a folder (or set parents)
  cp              copy a file
  rename          rename a file
  rm              trash or permanently delete
  restore         untrash (restore from trash)
  emptytrash      permanently delete everything in trash
  share           grant access (email / anyone / domain)
  unshare         revoke a permission
  perms           list permissions
  revs            list revisions
  star / unstar   toggle starred
  links           get shareable link (optionally create anyone-with-link)

Examples:
  gdrive about
  gdrive list --folder <id> --max 50
  gdrive search "quarterly report" --mime application/pdf
  gdrive find "Reports/Q4"           # -> id
  gdrive tree <folder_id> --depth 3
  gdrive upload ./report.pdf --parent <folder_id>
  gdrive import ./data.csv --as sheets
  gdrive download <file_id> --out ./report.pdf
  gdrive export <doc_id> --fmt pdf --out ./report.pdf
  gdrive mkdir "Q4" --parent <folder_id>
  gdrive share <file_id> --email a@b.c --role writer
  gdrive links <file_id> --anyone
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_HERMES = Path.home() / ".hermes"
TOKEN_PATH = Path(os.environ.get("GDRIVE_TOKEN", DEFAULT_HERMES / "google_token.json"))
SECRET_PATH = Path(os.environ.get("GDRIVE_SECRET", DEFAULT_HERMES / "google_client_secret.json"))
OUT_DIR = Path(os.environ.get("GDRIVE_OUT", DEFAULT_HERMES / "gdrive_output"))

SCOPE_DRIVE = "https://www.googleapis.com/auth/drive"
SCOPE_DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"

# gcloud ADC path — used when no OAuth token/client secret exists yet.
# The gcloud CLI OAuth client whitelists drive.file (app-scoped: files the
# app creates), which enables out-of-the-box operation on any gcloud
# authenticated machine. Full-drive scope requires the Desktop OAuth client.
ADC_PATH = Path(os.environ.get("GDRIVE_ADC", Path.home() / ".config/gcloud/application_default_credentials.json"))
QUOTA_PROJECT = os.environ.get("GDRIVE_QUOTA_PROJECT", "")

# MIME helpers
MIME_FOLDER = "application/vnd.google-apps.folder"
MIME_DOC = "application/vnd.google-apps.document"
MIME_SHEET = "application/vnd.google-apps.spreadsheet"
MIME_SLIDE = "application/vnd.google-apps.presentation"
MIME_DRAW = "application/vnd.google-apps.drawing"
MIME_SHORTCUT = "application/vnd.google-apps.shortcut"

# Import: local extension -> target Google mimeType
IMPORT_MAP = {
    "application/vnd.google-apps.document": ["docx", "doc", "odt", "rtf", "txt", "html"],
    "application/vnd.google-apps.spreadsheet": ["xlsx", "xls", "csv", "tsv", "ods"],
    "application/vnd.google-apps.presentation": ["pptx", "ppt", "odp"],
    "application/vnd.google-apps.drawing": ["png", "jpeg", "jpg", "svg"],
}
IMPORT_EXT_TO_MIME = {ext: m for m, exts in IMPORT_MAP.items() for ext in exts}

# Export: Google mimeType -> allowed export MIME types
EXPORT_MAP = {
    MIME_DOC: {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
        "html": "text/html",
        "rtf": "application/rtf",
        "odt": "application/vnd.oasis.opendocument.text",
        "md": "text/markdown",
    },
    MIME_SHEET: {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "tsv": "text/tab-separated-values",
        "ods": "application/x-vnd.oasis.opendocument.spreadsheet",
        "pdf": "application/pdf",
    },
    MIME_SLIDE: {
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pdf": "application/pdf",
        "txt": "text/plain",
    },
    MIME_DRAW: {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
    },
}
DEFAULT_EXPORT = {"pdf": "pdf", "docx": "docx", "xlsx": "xlsx", "csv": "csv", "pptx": "pptx", "png": "png"}

VALID_ROLES = ("reader", "writer", "commenter", "owner", "organizer")
VALID_PERM_TYPES = ("user", "group", "domain", "anyone")

# Order-by whitelist (safe to pass through)
VALID_ORDER_BY = (
    "createdTime", "folder", "modifiedByMeTime", "modifiedTime",
    "name", "name_natural", "quotaBytesUsed", "recency",
    "sharedWithMeTime", "starred", "viewedByMeTime",
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def out(data):
    """Print compact JSON to stdout (single line, no trailing spaces)."""
    sys.stdout.write(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.write("\n")


def fail(msg, code=1):
    """Structured error: JSON on stdout, non-zero exit."""
    out({"e": msg})
    sys.exit(code)


def pointer(path, n=None, **extra):
    """Write full payload to disk, print a small pointer on stdout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(n, (list, dict)):
        # payload directly
        payload = n
        n = len(payload) if isinstance(payload, list) else None
    else:
        payload = n
    path.write_text(
        json.dumps(payload if payload is not None else {}, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    d = {"ok": True, "f": str(path)}
    if isinstance(n, int):
        d["n"] = n
    d.update(extra)
    out(d)


def log(msg):
    """Human status -> stderr (never stdout, keeps JSON clean)."""
    sys.stderr.write(msg + "\n")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_secret():
    if not SECRET_PATH.exists():
        fail(
            f"No OAuth client secret at {SECRET_PATH}. Run `gdrive auth` for the setup flow, "
            "or set GDRIVE_SECRET to your client_secret_*.json path."
        )
    try:
        return json.loads(SECRET_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        fail(f"Client secret file {SECRET_PATH} is not valid JSON.")


def get_credentials():
    """Load token, refresh if needed. Falls back to gcloud ADC (drive.file)
    when no OAuth token exists yet. Returns google.oauth2.credentials.Credentials."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not TOKEN_PATH.exists():
        return _adc_credentials()

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    except Exception as e:
        fail(f"Token file {TOKEN_PATH} is corrupt: {e}")
    if creds and creds.expired and creds.refresh_token:
        log("Token expired, refreshing...")
        try:
            creds.refresh(Request())
        except Exception as e:
            fail(f"Token refresh failed ({e}). Run `gdrive auth` again to re-authorize.")
        _save_token(creds)
    elif creds and not creds.valid:
        fail("Token is invalid. Run `gdrive auth` again.")
    return creds


def _adc_credentials():
    """gcloud Application Default Credentials -> Drive credentials.

    Uses drive.file scope (whitelisted for the gcloud OAuth client): full
    operation on files the app creates. Returns None-friendly fail JSON if
    no ADC found.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not ADC_PATH.exists():
        fail(
            f"No OAuth token at {TOKEN_PATH} and no gcloud ADC at {ADC_PATH}. "
            "Run `gdrive auth` (one-time browser approval) or `gcloud auth application-default login`."
        )
    try:
        data = json.loads(ADC_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        fail(f"gcloud ADC file {ADC_PATH} is not valid JSON.")
    if data.get("type") != "authorized_user":
        fail(f"gcloud ADC at {ADC_PATH} is type '{data.get('type')}' — only 'authorized_user' ADC is supported. Use `gdrive auth` for the OAuth flow.")
    quota_project = QUOTA_PROJECT
    if not quota_project:
        # Best-effort: discover the user's first GCP project for quota.
        try:
            quota_project = _discover_quota_project()
        except Exception:
            quota_project = ""
    try:
        kwargs = {}
        if quota_project:
            kwargs["quota_project_id"] = quota_project
        creds = Credentials(
            token=None,
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=[SCOPE_DRIVE_FILE],
            **kwargs,
        )
        creds.refresh(Request())
    except Exception as e:
        fail(f"gcloud ADC refresh failed ({e}). Run `gdrive auth` or `gcloud auth application-default login`.")
    return creds


def _discover_quota_project():
    """Return the user's first GCP project id (for x-goog-user-project).

    Uses the ADC itself to list projects via cloudresourcemanager. Returns ''
    when none found; never raises.
    """
    import urllib.request
    import urllib.parse

    try:
        data = json.loads(ADC_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_info(
            data, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        req = urllib.request.Request(
            "https://cloudresourcemanager.googleapis.com/v1/projects",
            headers={"Authorization": "Bearer " + creds.token},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        projects = resp.get("projects") or []
        for p in projects:
            if p.get("lifecycleState") == "ACTIVE":
                return p["projectId"]
    except Exception:
        return ""
    return ""


def _save_token(creds):
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    TOKEN_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(TOKEN_PATH, 0o600)


def cmd_auth(args):
    if args.action == "revoke":
        if TOKEN_PATH.exists():
            TOKEN_PATH.unlink()
            out({"ok": True, "m": "token revoked"})
        else:
            out({"ok": True, "m": "no token present"})
        return

    secret = load_secret() if SECRET_PATH.exists() else None

    if args.action == "status":
        if not TOKEN_PATH.exists() and not ADC_PATH.exists():
            out({"ok": False, "auth": False, "m": "not authenticated"})
            return
        try:
            creds = get_credentials()
            about = _service().about().get(fields="user(emailAddress,displayName)").execute()
            src = "oauth" if TOKEN_PATH.exists() else "gcloud_adc"
            out({
                "ok": True,
                "auth": True,
                "src": src,
                "email": about.get("user", {}).get("emailAddress", "?"),
                "name": about.get("user", {}).get("displayName", "?"),
            })
        except Exception as e:
            out({"ok": False, "auth": False, "e": str(e)})
        return

    # -- full setup flow (PKCE) --
    if secret is None:
        fail(
            f"Missing client secret at {SECRET_PATH}.\n"
            "Create one (5 min):\n"
            "  1. https://console.cloud.google.com/apis/credentials\n"
            "  2. Create Credentials -> OAuth 2.0 Client ID -> Desktop app -> Create\n"
            "  3. Download the JSON and save it as:\n"
            f"     {SECRET_PATH}\n"
            "  If the project is in Testing, add your Google account as a test user:\n"
            "  https://console.cloud.google.com/auth/audience\n"
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    client_id = secret.get("installed", secret).get("client_id")
    if not client_id:
        fail("Client secret file has no 'installed' section (expected a Desktop app JSON).")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(SECRET_PATH), scopes=[SCOPE_DRIVE]
    )
    flow.redirect_uri = "http://localhost:1"

    # Build auth URL ourselves so we can print it for the user to open.
    # (InstalledAppFlow.run_local_server may not work headless.)
    import urllib.parse
    verifier = flow.oauth2session._client.create_code_verifier()
    challenge = flow.oauth2session._client.create_code_challenge(verifier)
    flow.oauth2session._client.code_verifier = verifier
    params = {
        "client_id": client_id,
        "redirect_uri": "http://localhost:1",
        "response_type": "code",
        "scope": SCOPE_DRIVE,
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = flow.oauth2session.authorization_url(**params)[0]

    log("=" * 62)
    log("Open this URL in a browser and approve access (pick the right account):")
    log("")
    log(auth_url)
    log("")
    log("After approving, the browser will show a connection error page")
    log("(http://localhost:1/?code=...). Copy the ENTIRE URL from the address bar")
    log("and paste it below.")
    log("=" * 62)

    redirect = input("Paste the redirect URL (or just the code): ").strip()
    if not redirect:
        fail("No code provided; auth aborted.")
    if "code=" not in redirect:
        # treat as bare code
        redirect = "http://localhost:1/?code=" + redirect

    code = urllib.parse.parse_qs(urllib.parse.urlparse(redirect).query).get("code", [None])[0]
    if not code:
        fail("Could not extract the code from that URL.")
    try:
        flow.fetch_token(code=code)
    except Exception as e:
        fail(f"Token exchange failed: {e}. The code may have expired — rerun `gdrive auth`.")

    creds = flow.credentials
    _save_token(creds)
    out({"ok": True, "auth": True, "m": "authenticated", "email": None})


def _service():
    """Build an authorized Drive v3 service (auto-refreshes)."""
    from googleapiclient.discovery import build

    creds = get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Field / payload shaping (short keys for tokens)
# ---------------------------------------------------------------------------

def _file(f):
    """Compact metadata dict for a Drive file."""
    d = {"id": f.get("id"), "n": f.get("name"), "m": f.get("mimeType")}
    if f.get("size") is not None:
        d["sz"] = int(f["size"])
    if f.get("modifiedTime"):
        d["t"] = f["modifiedTime"]
    if f.get("starred"):
        d["st"] = True
    if f.get("trashed"):
        d["tr"] = True
    if f.get("parents"):
        d["p"] = f["parents"]
    if f.get("webViewLink"):
        d["u"] = f["webViewLink"]
    if f.get("webContentLink"):
        d["dl"] = f["webContentLink"]
    if f.get("createdTime"):
        d["c"] = f["createdTime"]
    if f.get("owners"):
        d["o"] = [o.get("emailAddress") for o in f.get("owners", [])]
    if f.get("shortcutDetails"):
        d["target"] = f["shortcutDetails"].get("targetId")
    if f.get("description"):
        d["d"] = f["description"]
    return d


_FIELDS = ("id,name,mimeType,size,modifiedTime,createdTime,starred,trashed,parents,"
           "webViewLink,webContentLink,owners(emailAddress),shortcutDetails,description")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def cmd_about(args):
    svc = _service()
    a = svc.about().get(fields="user(emailAddress,displayName),storageQuota,driveThemes").execute()
    q = a.get("storageQuota", {})
    limit = int(q.get("limit", 0))
    used = int(q.get("usage", 0))
    d = {
        "email": a.get("user", {}).get("emailAddress"),
        "name": a.get("user", {}).get("displayName"),
        "quota": {"lim": limit, "used": used, "rem": max(limit - used, 0)},
    }
    if limit:
        d["quota"]["pct"] = round(100.0 * used / limit, 1)
    out({"ok": True, **d})


def _query(args):
    """Build a Drive files.list query string from filters."""
    parts = []
    if getattr(args, "query", None):
        if getattr(args, "raw_query", False):
            parts.append(args.query)
        else:
            q = args.query.replace("'", "\\'")
            parts.append(f"fullText contains '{q}'")
    if getattr(args, "folder", None):
        parts.append(f"'{args.folder}' in parents")
    if getattr(args, "mime", None):
        m = args.mime
        if m == "folder":
            m = MIME_FOLDER
        elif m == "doc":
            m = MIME_DOC
        elif m == "sheet":
            m = MIME_SHEET
        elif m == "slide":
            m = MIME_SLIDE
        parts.append(f"mimeType = '{m}'")
    if not getattr(args, "trashed", False):
        parts.append("trashed = false")
    else:
        parts.append("trashed = true")
    return " and ".join(parts)


def _list_files(svc, q, max_n, order_by=None, page_token=None):
    req = {"q": q, "pageSize": min(max_n, 1000), "fields": f"nextPageToken,files({_FIELDS})", "spaces": "drive"}
    if order_by:
        req["orderBy"] = order_by
    if page_token:
        req["pageToken"] = page_token
    return svc.files().list(**req).execute()


def cmd_list(args):
    order_by = None
    if getattr(args, "order", None):
        order_by = ",".join(args.order)
        for o in args.order:
            base = o.replace(" desc", "").replace(" asc", "")
            if base not in VALID_ORDER_BY:
                fail(f"Invalid orderBy term '{base}'. Valid: {', '.join(VALID_ORDER_BY)}")
    svc = _service()
    q = _query(args)
    files, token, pages = [], None, 0
    while True:
        res = _list_files(svc, q, args.max, order_by, token)
        files.extend(res.get("files", []))
        token = res.get("nextPageToken")
        pages += 1
        if not token or len(files) >= args.max or pages >= 100:
            break
    files = files[: args.max]
    if args.out:
        pointer(args.out, [ _file(f) for f in files ])
        return
    out({"ok": True, "n": len(files), "items": [_file(f) for f in files]})


def cmd_search(args):
    # alias for list with query
    args.query = " ".join(args.terms)
    cmd_list(args)


def cmd_get(args):
    svc = _service()
    try:
        f = svc.files().get(fileId=args.id, fields=_FIELDS).execute()
    except Exception as e:
        fail(f"get failed: {_apierr(e)}")
    out({"ok": True, **_file(f)})


def cmd_find(args):
    """Resolve a path like 'a/b/c' to an id. Paths are relative to My Drive root."""
    svc = _service()
    parts = [p for p in args.path.split("/") if p and p not in (".", "..")]
    if not parts:
        out({"ok": True, "id": "root", "n": "My Drive"})
        return
    cur = "root"
    for i, name in enumerate(parts):
        q = f"'{cur}' in parents and name = '{name.replace(chr(39), chr(92) + chr(39))}' and trashed = false"
        res = _list_files(svc, q, 10)
        files = res.get("files", [])
        if not files:
            fail(f"Not found: {'/'.join(parts[:i+1])} (no child named '{name}' under {cur})")
        if len(files) > 1:
            # disambiguate: prefer folders, then exact mime match, else first
            folders = [f for f in files if f.get("mimeType") == MIME_FOLDER]
            cur = (folders or files)[0]["id"]
        else:
            cur = files[0]["id"]
    out({"ok": True, "id": cur, "path": "/".join(parts)})


def _walk(svc, folder_id, depth, max_depth, out_list, prefix=""):
    q = f"'{folder_id}' in parents and trashed = false"
    token = None
    while True:
        res = _list_files(svc, q, 1000, order_by="folder,name_natural", page_token=token)
        for f in res.get("files", []):
            is_dir = f.get("mimeType") == MIME_FOLDER
            out_list.append({
                "id": f["id"],
                "n": f.get("name"),
                "m": f.get("mimeType"),
                "dir": is_dir,
                "sz": int(f["size"]) if f.get("size") else None,
                "path": prefix + f.get("name", ""),
            })
            if is_dir and depth < max_depth:
                _walk(svc, f["id"], depth + 1, max_depth, out_list, prefix + f.get("name", "") + "/")
        token = res.get("nextPageToken")
        if not token:
            break


def cmd_tree(args):
    svc = _service()
    root = args.id or "root"
    items = []
    _walk(svc, root, 1, args.depth, items)
    out_file = args.out or (OUT_DIR / f"tree_{root}_{int(time.time())}.json")
    pointer(out_file, items)


def _resolve_target(svc, target):
    """Accept an id or a path (contains '/') or a name; return fileId."""
    if not target:
        return "root"
    if "/" in target:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                cmd_find(argparse.Namespace(path=target))
            except SystemExit as e:
                if e.code != 0:
                    raise
        data = json.loads(buf.getvalue())
        return data["id"]
    return target


def _guess_mime(path):
    import mimetypes
    m, _ = mimetypes.guess_type(str(path))
    return m or "application/octet-stream"


def cmd_upload(args):
    from googleapiclient.http import MediaFileUpload

    path = Path(args.file)
    if not path.exists():
        fail(f"Local file not found: {path}")
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    name = args.name or path.name
    mime = args.mime or _guess_mime(path)
    size = path.stat().st_size
    media = MediaFileUpload(str(path), mimetype=mime, resumable=size > 8 * 1024 * 1024, chunksize=8 * 1024 * 1024)
    body = {"name": name, "mimeType": mime, "parents": [parent]} if parent and parent != "root" else {"name": name, "mimeType": mime}
    request = svc.files().create(body=body, media_body=media, fields=_FIELDS, supportsAllDrives=True)
    f = _execute_upload(request, media)
    out({"ok": True, **_file(f)})


def _execute_upload(request, media=None):
    from googleapiclient.errors import HttpError

    # Non-resumable (small) uploads respond to execute(), not next_chunk().
    is_resumable = False
    if media is not None:
        r = getattr(media, "resumable", None)
        is_resumable = bool(r()) if callable(r) else bool(r)
    if not is_resumable:
        try:
            return request.execute()
        except HttpError as e:
            fail(f"upload failed: {_apierr(e)}")
        except Exception as e:
            fail(f"upload error: {e}")
    resp = None
    while resp is None:
        try:
            _, resp = request.next_chunk()
        except HttpError as e:
            fail(f"upload failed: {_apierr(e)}")
        except Exception as e:
            fail(f"upload error: {e}")
    return resp


def cmd_import(args):
    from googleapiclient.http import MediaFileUpload

    path = Path(args.file)
    if not path.exists():
        fail(f"Local file not found: {path}")
    ext = path.suffix.lower().lstrip(".")
    target = args.as_type or IMPORT_EXT_TO_MIME.get(ext)
    if not target:
        fail(
            f"Don't know how to import '{ext}' files. Supported: "
            + ", ".join(sorted(set(IMPORT_EXT_TO_MIME)))
            + ". Use --as doc|sheet|slide|draw."
        )
    aliases = {"doc": MIME_DOC, "sheet": MIME_SHEET, "slide": MIME_SLIDE, "draw": MIME_DRAW}
    target = aliases.get(target, target)
    if target not in IMPORT_MAP:
        fail(f"Unknown target type '{target}'. Use --as doc|sheet|slide|draw.")
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    mime = _guess_mime(path)
    media = MediaFileUpload(str(path), mimetype=mime, resumable=path.stat().st_size > 8 * 1024 * 1024)
    body = {"name": args.name or path.stem, "mimeType": target}
    if parent and parent != "root":
        body["parents"] = [parent]
    request = svc.files().create(body=body, media_body=media, fields=_FIELDS, supportsAllDrives=True)
    f = _execute_upload(request, media)
    out({"ok": True, "conv": target, **_file(f)})


def cmd_download(args):
    import io

    svc = _service()
    f = svc.files().get(fileId=args.id, fields="id,name,mimeType,size,exportLinks").execute()
    mime = f.get("mimeType", "")
    out_path = Path(args.out) if args.out else Path.cwd() / f.get("name", args.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mime in EXPORT_MAP:
        fmt = args.fmt or "pdf"
        if fmt not in EXPORT_MAP[mime]:
            fail(f"'{fmt}' not exportable for {mime}. Allowed: {', '.join(EXPORT_MAP[mime])}")
        mime_out = EXPORT_MAP[mime][fmt]
        if not out_path.suffix:
            out_path = out_path.with_suffix("." + fmt)
        request = svc.files().export_media(fileId=args.id, mimeType=mime_out)
        _media_to_file(request, out_path)
        out({"ok": True, "f": str(out_path), "m": mime_out, "sz": out_path.stat().st_size})
        return

    # binary download
    request = svc.files().get_media(fileId=args.id)
    _media_to_file(request, out_path)
    out({"ok": True, "f": str(out_path), "sz": out_path.stat().st_size})


def _media_to_file(request, out_path):
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError

    fh = out_path.open("wb")
    try:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            try:
                status, done = downloader.next_chunk()
            except HttpError as e:
                fail(f"download failed: {_apierr(e)}")
    finally:
        fh.close()


def cmd_export(args):
    # export == download with explicit fmt; reuse
    args.fmt = args.fmt or "pdf"
    args.out = args.out
    cmd_download(args)


def cmd_mkdir(args):
    if not args.name or not args.name.strip():
        fail("folder name cannot be empty")
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    body = {"name": args.name, "mimeType": MIME_FOLDER}
    if parent and parent != "root":
        body["parents"] = [parent]
    f = svc.files().create(body=body, fields="id,name,mimeType,webViewLink,parents").execute()
    out({"ok": True, "id": f["id"], "n": f["name"], "m": f["mimeType"], "u": f.get("webViewLink")})


def cmd_touch(args):
    if not args.name or not args.name.strip():
        fail("file name cannot be empty")
    kind = args.kind or "doc"
    mime = {"doc": MIME_DOC, "sheet": MIME_SHEET, "slide": MIME_SLIDE}.get(kind)
    if not mime:
        fail("--kind must be doc|sheet|slide")
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    body = {"name": args.name, "mimeType": mime}
    if parent and parent != "root":
        body["parents"] = [parent]
    f = svc.files().create(body=body, fields=_FIELDS).execute()
    out({"ok": True, **_file(f)})


def cmd_mv(args):
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    if parent == "root":
        fail("mv needs a destination folder (--parent), not root.")
    out_list = []
    for fid in args.ids:
        try:
            f = svc.files().get(fileId=fid, fields="id,name,parents").execute()
            old_parents = ",".join(f.get("parents", []))
            svc.files().update(
                fileId=fid,
                addParents=parent,
                removeParents=old_parents,
                fields="id,parents",
                supportsAllDrives=True,
            ).execute()
            out_list.append({"id": fid, "n": f.get("name"), "to": parent})
        except Exception as e:
            out_list.append({"id": fid, "e": _apierr(e)})
    out({"ok": True, "n": len(out_list), "moved": out_list})


def cmd_cp(args):
    svc = _service()
    parent = _resolve_target(svc, args.parent)
    body = {}
    if args.name:
        body["name"] = args.name
    if parent and parent != "root":
        body["parents"] = [parent]
    try:
        f = svc.files().copy(fileId=args.id, body=body, fields=_FIELDS, supportsAllDrives=True).execute()
    except Exception as e:
        fail(f"copy failed: {_apierr(e)}")
    out({"ok": True, **_file(f)})


def cmd_rename(args):
    svc = _service()
    try:
        f = svc.files().update(fileId=args.id, body={"name": args.name}, fields="id,name,modifiedTime").execute()
    except Exception as e:
        fail(f"rename failed: {_apierr(e)}")
    out({"ok": True, "id": f["id"], "n": f["name"]})


def cmd_rm(args):
    svc = _service()
    results = []
    for fid in args.ids:
        try:
            if args.permanent:
                svc.files().delete(fileId=fid, supportsAllDrives=True).execute()
                results.append({"id": fid, "del": True})
            else:
                svc.files().update(fileId=fid, body={"trashed": True}, fields="id,trashed").execute()
                results.append({"id": fid, "tr": True})
        except Exception as e:
            results.append({"id": fid, "e": _apierr(e)})
    out({"ok": True, "n": len(results), "items": results})


def cmd_restore(args):
    svc = _service()
    results = []
    for fid in args.ids:
        try:
            svc.files().update(fileId=fid, body={"trashed": False}, fields="id,trashed").execute()
            results.append({"id": fid, "tr": False})
        except Exception as e:
            results.append({"id": fid, "e": _apierr(e)})
    out({"ok": True, "n": len(results), "items": results})


def cmd_emptytrash(args):
    svc = _service()
    svc.files().emptyTrash().execute()
    out({"ok": True, "m": "trash emptied"})


def cmd_share(args):
    svc = _service()
    body = {"role": args.role, "type": args.type}
    if args.email:
        body["emailAddress"] = args.email
    if args.domain:
        body["domain"] = args.domain
    try:
        p = svc.permissions().create(
            fileId=args.id, body=body,
            sendNotificationEmail=args.notify,
            emailMessage=args.message,
            supportsAllDrives=True,
            fields="id,role,type,emailAddress,domain,expirationTime",
        ).execute()
    except Exception as e:
        fail(f"share failed: {_apierr(e)}")
    out({"ok": True, "perm": p.get("id"), "role": p.get("role"), "type": p.get("type"),
         "email": p.get("emailAddress"), "domain": p.get("domain")})


def cmd_unshare(args):
    svc = _service()
    try:
        svc.permissions().delete(fileId=args.id, permissionId=args.perm, supportsAllDrives=True).execute()
    except Exception as e:
        fail(f"unshare failed: {_apierr(e)}")
    out({"ok": True, "m": "permission removed", "perm": args.perm, "id": args.id})


def cmd_perms(args):
    svc = _service()
    try:
        res = svc.permissions().list(
            fileId=args.id, fields="permissions(id,role,type,emailAddress,domain,expirationTime)",
            supportsAllDrives=True,
        ).execute()
    except Exception as e:
        fail(f"perms failed: {_apierr(e)}")
    perms = [
        {"id": p.get("id"), "role": p.get("role"), "type": p.get("type"),
         "email": p.get("emailAddress"), "domain": p.get("domain")}
        for p in res.get("permissions", [])
    ]
    out({"ok": True, "n": len(perms), "items": perms})


def cmd_revs(args):
    svc = _service()
    try:
        res = svc.revisions().list(
            fileId=args.id,
            fields="revisions(id,modifiedTime,size,keepForever,originalFilename,mimeType)",
        ).execute()
    except Exception as e:
        fail(f"revs failed: {_apierr(e)}")
    revs = [
        {"id": r.get("id"), "t": r.get("modifiedTime"), "sz": int(r["size"]) if r.get("size") else None,
         "keep": r.get("keepForever"), "fn": r.get("originalFilename"), "m": r.get("mimeType")}
        for r in res.get("revisions", [])
    ]
    out({"ok": True, "n": len(revs), "items": revs})


def cmd_star(args):
    svc = _service()
    results = []
    for fid in args.ids:
        try:
            svc.files().update(fileId=fid, body={"starred": True}, fields="id,starred").execute()
            results.append({"id": fid, "st": True})
        except Exception as e:
            results.append({"id": fid, "e": _apierr(e)})
    out({"ok": True, "n": len(results), "items": results})


def cmd_unstar(args):
    svc = _service()
    results = []
    for fid in args.ids:
        try:
            svc.files().update(fileId=fid, body={"starred": False}, fields="id,starred").execute()
            results.append({"id": fid, "st": False})
        except Exception as e:
            results.append({"id": fid, "e": _apierr(e)})
    out({"ok": True, "n": len(results), "items": results})


def cmd_links(args):
    svc = _service()
    f = svc.files().get(fileId=args.id, fields="id,name,mimeType,webViewLink").execute()
    if args.anyone:
        svc.permissions().create(
            fileId=args.id, body={"role": args.role, "type": "anyone"}, supportsAllDrives=True
        ).execute()
        f = svc.files().get(fileId=args.id, fields="id,name,webViewLink").execute()
    d = {"ok": True, "id": f["id"], "n": f.get("name"), "view": f.get("webViewLink")}
    if args.anyone:
        d["anyone"] = True
    out(d)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _apierr(e):
    """Human-readable API error."""
    from googleapiclient.errors import HttpError

    if isinstance(e, HttpError):
        try:
            body = json.loads(e.content.decode("utf-8", errors="replace"))
            return body.get("error", {}).get("message", str(e))
        except Exception:
            return str(e)
    return str(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="gdrive",
        description="AI-agent-native token-efficient Google Drive CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Actions:")[1].split("Examples:")[0] if "Actions:" in __doc__ else "",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, ids=True, multiple_ids=False):
        if ids:
            if multiple_ids:
                sp.add_argument("ids", nargs="+", help="one or more file/folder IDs")
            else:
                sp.add_argument("id", help="file or folder ID")
        sp.add_argument("--out", help="write full payload to this file (pointer on stdout)")

    # auth
    a = sub.add_parser("auth", help="OAuth2 setup / status / revoke")
    a.add_argument("action", nargs="?", default="setup", choices=["setup", "status", "revoke"])

    # about
    a = sub.add_parser("about", help="account + storage quota")
    a.add_argument("--out", help="save payload to file")

    # list
    a = sub.add_parser("list", help="list files with filters")
    a.add_argument("--query", "-q", help="full-text search term (or raw Drive query with --raw)")
    a.add_argument("--raw", action="store_true", dest="raw_query", help="treat --query as raw Drive query")
    a.add_argument("--folder", help="list children of this folder id/path")
    a.add_argument("--mime", help="filter mime (folder|doc|sheet|slide or full mimeType)")
    a.add_argument("--trashed", action="store_true", help="list trashed instead of active")
    a.add_argument("--max", type=int, default=50, help="max results (default 50)")
    a.add_argument("--order", action="append", help="orderBy term, repeatable (name, modifiedTime desc, ...)")
    a.add_argument("--out", help="save payload to file")

    # search
    a = sub.add_parser("search", help="full-text search (alias for list)")
    a.add_argument("terms", nargs="+", help="search terms")
    a.add_argument("--mime", help="filter mime")
    a.add_argument("--max", type=int, default=20)
    a.add_argument("--trashed", action="store_true")
    a.add_argument("--order", action="append")
    a.add_argument("--out", help="save payload to file")

    # get
    a = sub.add_parser("get", help="metadata for one file")
    add_common(a)
    a.add_argument("--path", action="store_true", help="treat id as a path like a/b/c")

    # find
    a = sub.add_parser("find", help="resolve a path to an id")
    a.add_argument("path", help="path like 'Reports/Q4' (relative to My Drive)")

    # tree
    a = sub.add_parser("tree", help="recursive folder listing")
    a.add_argument("id", nargs="?", default="root", help="folder id or path (default root)")
    a.add_argument("--depth", type=int, default=3, help="max depth (default 3)")
    a.add_argument("--out", help="save payload to file")

    # upload
    a = sub.add_parser("upload", help="upload a local file")
    a.add_argument("file", help="local file path")
    a.add_argument("--parent", help="destination folder id or path")
    a.add_argument("--name", help="name on Drive (default: basename)")
    a.add_argument("--mime", help="override MIME type")

    # import
    a = sub.add_parser("import", help="upload + convert to Google format")
    a.add_argument("file", help="local file path")
    a.add_argument("--as", dest="as_type", help="doc|sheet|slide|draw (default: by extension)")
    a.add_argument("--parent", help="destination folder id or path")
    a.add_argument("--name", help="name on Drive (default: stem)")

    # download / export
    a = sub.add_parser("download", help="download a binary file")
    a.add_argument("id", help="file id")
    a.add_argument("--out", help="output path (default: cwd/name)")
    a.add_argument("--fmt", help="export format for Google files (default pdf)")
    a = sub.add_parser("export", help="export a Google-native file")
    a.add_argument("id", help="file id")
    a.add_argument("--fmt", default="pdf", help="pdf|docx|txt|html|rtf|odt|md (doc), xlsx|csv|tsv|ods (sheet), pptx|pdf|txt (slide), png|jpeg|svg|pdf (draw)")
    a.add_argument("--out", help="output path")

    # mkdir / touch
    a = sub.add_parser("mkdir", help="create a folder")
    a.add_argument("name", help="folder name")
    a.add_argument("--parent", help="parent folder id or path")
    a = sub.add_parser("touch", help="create an empty Google file")
    a.add_argument("name", help="file name")
    a.add_argument("--kind", choices=["doc", "sheet", "slide"], default="doc")
    a.add_argument("--parent", help="parent folder id or path")

    # mv / cp / rename
    a = sub.add_parser("mv", help="move file(s) into a folder")
    a.add_argument("ids", nargs="+", help="file/folder IDs")
    a.add_argument("--parent", required=True, help="destination folder id or path")
    a = sub.add_parser("cp", help="copy a file")
    a.add_argument("id", help="file id")
    a.add_argument("--parent", help="destination folder id or path")
    a.add_argument("--name", help="new name")
    a = sub.add_parser("rename", help="rename a file")
    a.add_argument("id", help="file id")
    a.add_argument("name", help="new name")

    # rm / restore / emptytrash
    a = sub.add_parser("rm", help="trash (default) or permanently delete")
    a.add_argument("ids", nargs="+", help="file/folder IDs")
    a.add_argument("--permanent", action="store_true", help="skip trash, permanent delete")
    a = sub.add_parser("restore", help="restore from trash")
    a.add_argument("ids", nargs="+", help="file/folder IDs")
    a = sub.add_parser("emptytrash", help="permanently empty trash")

    # share / unshare / perms
    a = sub.add_parser("share", help="grant access")
    a.add_argument("id", help="file id")
    a.add_argument("--email", help="email address (type=user)")
    a.add_argument("--domain", help="domain (type=domain)")
    a.add_argument("--type", default="user", choices=VALID_PERM_TYPES)
    a.add_argument("--role", default="reader", choices=VALID_ROLES)
    a.add_argument("--notify", action="store_true", help="send notification email")
    a.add_argument("--message", help="notification message")
    a = sub.add_parser("unshare", help="revoke a permission")
    a.add_argument("id", help="file id")
    a.add_argument("perm", help="permission id (from perms)")
    a = sub.add_parser("perms", help="list permissions")
    a.add_argument("id", help="file id")

    # revs
    a = sub.add_parser("revs", help="list revisions")
    a.add_argument("id", help="file id")

    # star / unstar / links
    a = sub.add_parser("star", help="star file(s)")
    a.add_argument("ids", nargs="+")
    a = sub.add_parser("unstar", help="unstar file(s)")
    a.add_argument("ids", nargs="+")
    a = sub.add_parser("links", help="get shareable link")
    a.add_argument("id", help="file id")
    a.add_argument("--anyone", action="store_true", help="create 'anyone with link' permission")
    a.add_argument("--role", default="reader", choices=VALID_ROLES)

    return p


_HANDLERS = {
    "auth": cmd_auth,
    "about": cmd_about,
    "list": cmd_list,
    "search": cmd_search,
    "get": cmd_get,
    "find": cmd_find,
    "tree": cmd_tree,
    "upload": cmd_upload,
    "import": cmd_import,
    "download": cmd_download,
    "export": cmd_export,
    "mkdir": cmd_mkdir,
    "touch": cmd_touch,
    "mv": cmd_mv,
    "cp": cmd_cp,
    "rename": cmd_rename,
    "rm": cmd_rm,
    "restore": cmd_restore,
    "emptytrash": cmd_emptytrash,
    "share": cmd_share,
    "unshare": cmd_unshare,
    "perms": cmd_perms,
    "revs": cmd_revs,
    "star": cmd_star,
    "unstar": cmd_unstar,
    "links": cmd_links,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        _HANDLERS[args.command](args)
    except KeyboardInterrupt:
        fail("interrupted")
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{args.command} failed: {_apierr(e)}")


if __name__ == "__main__":
    main()
