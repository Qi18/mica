#!/usr/bin/env python3
"""Fast, dependency-free checks for the public repository boundary."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
DISALLOWED_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
SECRET_MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN RSA " + b"PRIVATE KEY-----",
    b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
    b"github_" + b"pat_",
    b"gh" + b"p_",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def local_link_target(markdown: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "mailto:")) or "://" in target:
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return (markdown.parent / target).resolve()


def main() -> None:
    failures: list[str] = []
    files = tracked_files()

    for path in files:
        relative = path.relative_to(ROOT)
        if not path.exists():
            failures.append(f"tracked path is missing: {relative}")
            continue
        if path.suffix.lower() in DISALLOWED_SUFFIXES or path.name.startswith(".env"):
            failures.append(f"generated or private file is tracked: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 10 MiB: {relative}")

        if path.stat().st_size <= MAX_TRACKED_BYTES:
            content = path.read_bytes()
            if any(marker in content for marker in SECRET_MARKERS):
                failures.append(f"possible credential marker: {relative}")

        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(text):
                target = local_link_target(path, match.group(1))
                if target is not None and not target.exists():
                    failures.append(
                        f"broken local link: {relative} -> {match.group(1)}"
                    )

    for required in (ROOT / "LICENSE", ROOT / "docs/upstream-minimind.md"):
        if not required.is_file():
            failures.append(f"missing upstream attribution asset: {required.relative_to(ROOT)}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    print(f"repository_check=pass tracked_files={len(files)}")


if __name__ == "__main__":
    main()
