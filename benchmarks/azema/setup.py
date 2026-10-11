"""Prepare isolated Python/Julia runtimes and immutable upstream source copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen
import venv
import zipfile

from . import COMMIT, GUROBI_VERSION, JULIA_VERSION, ROOT, RUNTIME


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "SUC-Azema-benchmark"}), timeout=120) as response:
        return response.read()


def prepare_sources():
    prefix = "Azema2026_Case_Inputs/"
    destination = RUNTIME / "author"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "Azema2026_Benders_Case_Inputs.zip") as archive:
        provenance = json.loads(archive.read(prefix + "provenance.json"))
        assert provenance["repository_commit"] == COMMIT
        for row in provenance["files"]:
            body = archive.read(prefix + row["path"])
            if len(body) != row["bytes"] or hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Input hash mismatch: {row['path']}")
            target = (destination / row["path"]).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Input path leaves the author directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
    tree = json.loads(fetch(f"https://api.github.com/repos/MathisAzema/BDSUC/git/trees/{COMMIT}?recursive=1"))
    blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    paths = [r["path"] for r in tree["tree"] if r["type"] == "blob" and
             (r["path"].startswith("src/") or r["path"] in ("README.md", "Project.toml", "Manifest.toml", "test.jl"))]
    hashes = {}
    for name in paths:
        target = destination / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fetch(f"https://raw.githubusercontent.com/MathisAzema/BDSUC/{COMMIT}/{name}"))
        body = target.read_bytes()
        git_hash = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if git_hash != blobs[name]:
            raise RuntimeError(f"Source does not match pinned Git blob: {name}")
        hashes[name] = hashlib.sha256(body).hexdigest()
    (RUNTIME / "source_manifest.json").write_text(json.dumps(dict(commit=COMMIT, files=hashes,
        git_blobs={name: blobs[name] for name in paths}, inputs=provenance["files"]), indent=2), encoding="utf-8")
    print(f"Verified 13 input files; pinned {len(paths)} upstream source files", flush=True)


def prepare_python():
    folder = RUNTIME / "python"
    python = folder / "Scripts" / "python.exe"
    if not python.exists():
        # Inherit the already licensed Kaiwu installation; override Gurobi only here.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(folder)
    subprocess.run([str(python), "-m", "pip", "install", f"gurobipy=={GUROBI_VERSION}",
                    "netCDF4==1.7.2", "tomli==2.2.1", "tomli-w==1.2.0"], check=True)
    return python


def prepare_julia():
    executable = RUNTIME / f"julia-{JULIA_VERSION}" / "bin" / "julia.exe"
    if not executable.exists():
        versions = json.loads(fetch("https://julialang-s3.julialang.org/bin/versions.json"))
        row = next(r for r in versions[JULIA_VERSION]["files"]
                   if r["triplet"] == "x86_64-w64-mingw32" and r["extension"] == "zip")
        package = RUNTIME / "julia.zip"
        if not package.exists() or hashlib.sha256(package.read_bytes()).hexdigest() != row["sha256"]:
            with urlopen(row["url"], timeout=120) as response, package.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if hashlib.sha256(package.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Julia distribution hash mismatch")
        with zipfile.ZipFile(package) as archive:
            for name in archive.namelist():
                if not (RUNTIME / name).resolve().is_relative_to(RUNTIME.resolve()):
                    raise ValueError("Julia archive path leaves runtime directory")
            archive.extractall(RUNTIME)
    environment = dict(os.environ, JULIA_DEPOT_PATH=str(RUNTIME / "julia-depot"),
                       JULIA_NUM_THREADS="1", JULIA_NUM_PRECOMPILE_TASKS="1", JULIA_PKG_PRECOMPILE_AUTO="0")
    # The pristine Project/Manifest remain alongside the source. Only the experiment
    # environment adds JSON3 for structured input/output; solver dependencies stay pinned.
    project = RUNTIME / "julia-env"
    project.mkdir(exist_ok=True)
    for name in ("Project.toml", "Manifest.toml"):
        if not (project / name).exists():
            shutil.copy2(RUNTIME / "author" / name, project / name)
    shutil.copytree(RUNTIME / "author" / "src", project / "src", dirs_exist_ok=True)
    script = 'using Pkg; Pkg.instantiate(;allow_autoprecomp=false); Pkg.add(PackageSpec(name="JSON3", version="1.14.3"); preserve=Pkg.PRESERVE_ALL);'
    subprocess.run([str(executable), f"--project={project}", "--startup-file=no", "-e", script],
                   env=environment, check=True)
    return executable


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use the existing Python 3.10 installation to retain Kaiwu compatibility")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    prepare_sources()
    python = prepare_python()
    julia = prepare_julia()
    (RUNTIME / "runtime.json").write_text(json.dumps(dict(python=str(python), julia=str(julia),
        julia_version=JULIA_VERSION, gurobi_version=GUROBI_VERSION, commit=COMMIT), indent=2), encoding="utf-8")
    print("Runtime preparation complete; no optimization or LLM calls were started", flush=True)


if __name__ == "__main__":
    main()
