"""Archive the audited recovery and retained attempts without changing their files."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

root = Path(__file__).resolve().parents[1]
base = root.parent
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if not summary["complete"] or not summary["numerical_audit_passed"]:
    raise SystemExit("finish and audit the batch before archiving")

folders = [base / name for name in (
    "rts24-recovery-20261004", "rts24-recovery-20261004-fixed",
    "rts24-recovery-20261004-account", root.name)]
files = sorted(p for folder in folders for p in folder.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
files += [base / "rts24-minimal-20261003.zip", base / "rts24-minimal-20261003.zip.sha256"]
archive = base / "rts24-recovery-20261004-bundle.zip"
records = {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in files}
checksums = "".join(f"{digest}  {name}\n" for name, digest in records.items())
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for path in files:
        output.write(path, path.relative_to(base).as_posix())
    output.writestr("SHA256SUMS.txt", checksums)

with zipfile.ZipFile(archive) as output:
    for name, expected in records.items():
        with output.open(name) as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise AssertionError(f"archive entry changed: {name}")
    if output.read("SHA256SUMS.txt").decode() != checksums:
        raise AssertionError("checksum manifest changed")
for path in files:
    if hashlib.sha256(path.read_bytes()).hexdigest() != records[path.relative_to(base).as_posix()]:
        raise AssertionError(f"source changed while archiving: {path}")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
verification = dict(archive=archive.name, sha256=digest, files_verified=len(files),
                    source_files_unchanged=True, verified_at=time.time())
archive.with_suffix(".zip.verification.json").write_text(
    json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(verification, ensure_ascii=False))
