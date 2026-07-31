"""Resumable download of the public sample ultratrace.

Uses ``aria2c`` (parallel HTTP range requests) when the binary is on the PATH:
multi-connection downloads hold up much better against the rate-limited r2.dev
endpoint. Falls back to a pure standard-library (``urllib``) sequential
downloader so the package still works with no extra installs. Both paths
support HTTP range resume so an interrupted ~96 GB download can be continued
in place.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

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
# Parallel connections for aria2c. The r2.dev endpoint throttles per
# connection, so a moderate fan-out helps; keep it polite.
_ARIA2_SPLITS = 8


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def _aria2_control_file(out: Path) -> Path:
    return out.with_name(out.name + ".aria2")


def _download_with_aria2(aria2c: str, url: str, out: Path) -> Path:
    command = [
        aria2c,
        "--continue=true",
        f"--split={_ARIA2_SPLITS}",
        f"--max-connection-per-server={_ARIA2_SPLITS}",
        "--min-split-size=16M",
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
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"aria2c exited with code {exc.returncode}. The partial file and its "
            f"{_aria2_control_file(out).name} control file are kept; re-running resumes."
        ) from exc
    print(f"Saved {out} ({_fmt_bytes(out.stat().st_size)})")
    return out


def _download_with_urllib(url: str, out: Path, total: int | None, chunk: int) -> Path:
    existing = out.stat().st_size if out.exists() else 0
    if total is not None and existing > total:
        # Local file is larger than remote -- assume stale, restart.
        out.unlink()
        existing = 0

    headers = {"User-Agent": _USER_AGENT}
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


def download_sample(
    url: str = SAMPLE_URL,
    output: str | Path = SAMPLE_FILENAME,
    *,
    force: bool = False,
    chunk: int = _CHUNK,
    downloader: str = "auto",
) -> Path:
    """Download ``url`` to ``output``, resuming a partial file when possible.

    ``downloader`` selects the backend: ``"auto"`` (default) uses ``aria2c``
    when the binary is on the PATH and urllib otherwise; ``"aria2"`` /
    ``"urllib"`` force one. Returns the resolved output path. Re-running after
    a complete download is a no-op unless ``force`` is set.
    """
    if downloader not in ("auto", "aria2", "urllib"):
        raise ValueError(
            f"Unknown downloader {downloader!r}; expected 'auto', 'aria2', or 'urllib'"
        )

    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    control = _aria2_control_file(out)

    if force:
        if out.exists():
            out.unlink()
        if control.exists():
            control.unlink()

    total = _remote_size(url)
    have_clean_file = out.exists() and not control.exists()
    if total is not None and have_clean_file and out.stat().st_size == total:
        print(f"Already complete: {out} ({_fmt_bytes(total)})")
        return out

    aria2c = shutil.which("aria2c")
    if downloader == "aria2" and aria2c is None:
        raise RuntimeError("downloader='aria2' but aria2c is not on the PATH")
    use_aria2 = downloader == "aria2" or (downloader == "auto" and aria2c is not None)

    if not use_aria2 and control.exists():
        # An aria2c partial is written in parallel segments and may contain
        # holes, so the sequential downloader must not append to it.
        raise RuntimeError(
            f"{out} was partially downloaded by aria2c ({control.name} exists) and "
            "may contain gaps the sequential urllib downloader cannot fill. "
            "Install aria2 to resume it, or pass --force to restart from scratch."
        )

    if use_aria2:
        print(f"Downloading with aria2c ({_ARIA2_SPLITS} connections) ...")
        return _download_with_aria2(aria2c, url, out)
    return _download_with_urllib(url, out, total=total, chunk=chunk)
