"""Build the final submission zip and its sha1 signature.

Run this once, last. Every invocation rebuilds the archive and rewrites
signature.txt; rebuild after the hash is recorded and the two no longer
match, which is an automatic zero.

The zip ships the code and the trained model/, never the raw data set.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

EXCLUDE_DIR_NAMES = {
    ".git", "images", "toy_dataset", "__pycache__", ".venv", ".pytest_cache",
}
EXCLUDE_SUFFIXES = {".zip"}
EXCLUDE_FILE_NAMES = {".DS_Store"}


def _should_include(relative_path: Path) -> bool:
    if any(part in EXCLUDE_DIR_NAMES for part in relative_path.parts):
        return False
    if relative_path.suffix in EXCLUDE_SUFFIXES:
        return False
    if relative_path.name in EXCLUDE_FILE_NAMES:
        return False
    return True


def build_zip(root: Path, zip_path: Path) -> Path:
    """Zip everything under root except the data set, the zip itself,
    and version control or cache directories."""
    root = Path(root)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _should_include(rel):
                zf.write(path, rel)
    return zip_path


def write_signature(zip_path: Path, signature_path: Path) -> str:
    """Write `<sha1>  <zip filename>` to signature_path and return the sha1."""
    zip_path = Path(zip_path)
    digest = hashlib.sha1(zip_path.read_bytes()).hexdigest()
    Path(signature_path).write_text(f"{digest}  {zip_path.name}\n")
    return digest


def build_release(
    root: Path = Path("."),
    zip_path: Path = Path("leaffliction.zip"),
    signature_path: Path = Path("signature.txt"),
) -> str:
    """Build the zip then hash it, in that order. Returns the sha1 digest."""
    build_zip(root, zip_path)
    return write_signature(zip_path, signature_path)


if __name__ == "__main__":
    digest = build_release()
    print(f"sha1 {digest}")
