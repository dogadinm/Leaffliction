import hashlib
import zipfile
from pathlib import Path

from leaffliction.package import build_release, build_zip, write_signature


def _make_repo(root: Path) -> None:
    (root / "train.py").write_text("print('train')\n")
    (root / "leaffliction").mkdir()
    (root / "leaffliction" / "io.py").write_text("# io\n")
    (root / "model").mkdir()
    (root / "model" / "model.keras").write_bytes(b"fake-weights")
    (root / "images").mkdir()
    (root / "images" / "leaked.jpg").write_bytes(b"should-not-ship")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "CACHEDIR.TAG").write_text("dev noise\n")
    (root / ".DS_Store").write_bytes(b"finder noise")


def test_build_zip_excludes_dataset_and_git(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    zip_path = tmp_path / "leaffliction.zip"

    build_zip(root, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    assert "train.py" in names
    assert "leaffliction/io.py" in names
    assert "model/model.keras" in names  # the deliverable includes the model
    assert not any(n.startswith("images/") for n in names)
    assert not any(n.startswith(".git/") for n in names)
    assert not any(n.startswith(".pytest_cache/") for n in names)
    assert ".DS_Store" not in names


def test_write_signature_matches_zip_sha1(tmp_path: Path) -> None:
    zip_path = tmp_path / "release.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "hello")
    sig_path = tmp_path / "signature.txt"

    digest = write_signature(zip_path, sig_path)

    expected = hashlib.sha1(zip_path.read_bytes()).hexdigest()
    assert digest == expected
    assert sig_path.read_text().startswith(expected)


def test_build_release_hash_reflects_the_zip_contents_at_build_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _make_repo(root)
    zip_path = tmp_path / "leaffliction.zip"
    sig_path = tmp_path / "signature.txt"

    digest = build_release(root, zip_path, sig_path)

    assert sig_path.read_text().split()[0] == digest
    assert digest == hashlib.sha1(zip_path.read_bytes()).hexdigest()
