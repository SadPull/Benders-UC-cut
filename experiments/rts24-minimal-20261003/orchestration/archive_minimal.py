"""Bundle the completed experiment and verify every archived byte."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

workspace = Path(r"C:\Users\12433\Desktop\LLM-benders-SUC")
root = workspace / "experiments/rts24-minimal-20261003"
summary = json.loads((root / "final-summary.json").read_text(encoding="utf-8"))
if not summary["passed"]:
    raise ValueError("the final numerical audit must pass before archiving")
helpers = root / "orchestration"
helpers.mkdir(exist_ok=True)
for name in ("resume_minimal.py", "replace_minimal.py", "watch_minimal.py", "finalize_minimal.py", "archive_minimal.py"):
    shutil.copy2(workspace / "results" / name, helpers / name)
for name in ("minimal-resumed.stdout.log", "minimal-resumed.stderr.log",
             "minimal-replacement.stdout.log", "minimal-replacement.stderr.log"):
    source = workspace / "results" / name
    if source.exists():
        shutil.copy2(source, helpers / name)
files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and p.suffix not in {".pyc", ".tmp"} and p.name != "SHA256SUMS.txt")
for p in files:
    if p.name.lower().startswith(".env") or p.suffix.lower() in {".pem", ".key", ".lic"}:
        raise ValueError(f"unexpected credential file in experiment: {p.name}")
digests = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
checksums = root / "SHA256SUMS.txt"
checksums.write_text("".join(f"{digest}  {name}\n" for name, digest in digests.items()), encoding="utf-8")
archive = root.with_suffix(".zip")
with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for p in files + [checksums]:
        bundle.write(p, (Path(root.name) / p.relative_to(root)).as_posix())
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for name, digest in digests.items():
        assert hashlib.sha256(bundle.read(f"{root.name}/{name}")).hexdigest() == digest, name
archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(f"{archive_digest}  {archive.name}\n", encoding="utf-8")
print(json.dumps(dict(archive=str(archive), files=len(files)+1, bytes=archive.stat().st_size,
                     sha256=archive_digest, verified=True)))
