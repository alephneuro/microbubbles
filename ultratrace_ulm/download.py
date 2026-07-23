"""Resumable download of the public sample ultratrace.
Pure standard library (``urllib``) so the package keeps its minimal dependency
footprint -- no ``curl``/``requests`` required. Supports HTTP range resume so an
interrupted ~96 GB download can be continued in place.

Uses aria2c when available for parallel HTTP range downloads.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

#: Sanitized neutral ultratrace (demodulated IQ + transmit delays + a
#: beamforming-only config) hosted on Cloudflare R2: ~96 GB, 216 acquisitions (Feb 2026 golden reference, 8-row elevation aperture).
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
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def _download_with_urllib(
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


def _download_with_aria2(
    url: str,
    output: str | Path,
    *,
    force: bool,
) -> Path:
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if force:
        out.unlink(missing_ok=True)
        out.with_name(f"{out.name}.aria2").unlink(missing_ok=True)

    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise RuntimeError(
            "aria2c is not installed.\n"
            "Install it on macOS with:\n"
            "  brew install aria2\n"
            "Or use:\n"
            "  --downloader urllib"
        )

    command = [
        aria2c,
        "--continue=true",
        "--split=8",
        "--max-connection-per-server=8",
        "--min-split-size=16M",
        "--max-tries=0",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=60",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        f"--user-agent={_USER_AGENT}",
        f"--dir={out.parent}",
        f"--out={out.name}",
        url,
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("aria2c executable was not found") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"aria2c failed with exit code {error.returncode}"
        ) from error

    if not out.is_file():
        raise RuntimeError(f"aria2c exited successfully but did not create {out}")

    remote_size = _remote_size(url)

    if remote_size is not None and out.stat().st_size != remote_size:
        raise RuntimeError(
            "aria2c exited but the file size is incorrect: "
            f"{_fmt_bytes(out.stat().st_size)} / {_fmt_bytes(remote_size)}"
        )

    print(f"Saved {out} ({_fmt_bytes(out.stat().st_size)})")
    return out


def download_sample(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    downloader: Literal["urllib", "aria2"] = "urllib",
) -> Path:
    """Download the sample file, resuming partial downloads when possible.

    Downloader modes:
    - ``aria2``: require aria2c and use parallel HTTP range requests
    - ``urllib``: use the Python standard-library sequential downloader
    """

    match downloader:
        case "aria2":
            print("Using aria2c downloader.")
            return _download_with_aria2(
                url,
                output,
                force=force,
            )
        case "urllib":
            return _download_with_urllib(
                url,
                output,
                force=force,
            )

    raise ValueError(f"Unsupported downloader {downloader!r}; ")
