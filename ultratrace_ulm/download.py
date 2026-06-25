"""Resumable download of the public sample ultratrace.

Pure standard library (``urllib``) so the package keeps its minimal dependency
footprint -- no ``curl``/``requests`` required. Supports HTTP range resume so an
interrupted ~98 GB download can be continued in place.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Sanitized neutral ultratrace (demodulated IQ + transmit delays + a
#: beamforming-only config) hosted on Cloudflare R2: ~98 GB, 223 acquisitions.
SAMPLE_URL = (
    "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/"
    "sanitized_neutral_ultratrace.h5"
)
SAMPLE_FILENAME = "sample_ultratrace.h5"

_CHUNK = 8 * 1024 * 1024  # 8 MiB
# Cloudflare R2's public endpoint returns 403 for urllib's default
# ``Python-urllib/x.y`` agent, so send an explicit one.
_USER_AGENT = "ultratrace-ulm/0.1"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def download_sample(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
) -> Path:
    """Download ``url`` to ``output``, resuming a partial file when possible.

    Returns the resolved output path. Re-running after a complete download is a
    no-op unless ``force`` is set.
    """
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)
    if force and out.exists():
        out.unlink()

    existing = out.stat().st_size if out.exists() else 0
    if total is not None and existing == total:
        print(f"Already complete: {out} ({_fmt_bytes(total)})")
        return out
    if total is not None and existing > total:
        # Local file is larger than remote -- assume stale, restart.
        out.unlink()
        existing = 0

    headers = {"User-Agent": _USER_AGENT}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {_fmt_bytes(existing)} ...")

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # Range Not Satisfiable -> already have it all.
            print(f"Already complete: {out} ({_fmt_bytes(existing)})")
            return out
        raise

    with resp:
        # Server honored the range request -> append; otherwise restart.
        if existing and resp.status == 206:
            mode = "ab"
            done = existing
        else:
            mode = "wb"
            done = 0
        grand_total = total
        cl = resp.headers.get("Content-Length")
        if grand_total is None and cl is not None:
            grand_total = done + int(cl)

        with open(out, mode) as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if grand_total:
                    pct = 100.0 * done / grand_total
                    bar = f"{_fmt_bytes(done)} / {_fmt_bytes(grand_total)} ({pct:.1f}%)"
                else:
                    bar = _fmt_bytes(done)
                print(f"\r  {bar}        ", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    print(f"Saved {out} ({_fmt_bytes(out.stat().st_size)})")
    return out
