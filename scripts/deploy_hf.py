"""Deploy the API to a Hugging Face Docker Space.

    pip install huggingface_hub
    hf auth login                                   # once; paste a token with "write" access
    python scripts/deploy_hf.py --user <your-hf-name>

Steps:
  1. Create (if necessary) the public dataset repository <user>/lunarsolve-data and upload the data
     files of server/fetch_data.py (about 1 GB; only changed files are sent again).
  2. Assemble deploy/build/ = deploy/hf-space/* + lunarsolve/ + server/ + weights/ + data_repo.txt.
  3. Create (if necessary) the Docker Space <user>/lunar-solver and upload deploy/build/.
     The Space then builds the image (about 5-10 minutes) and starts the API at
     https://<user>-lunar-solver.hf.space

Use --dry-run to only assemble deploy/build/ and print what would be uploaded.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.fetch_data import DATA_DIR, DATA_FILES  # noqa: E402


def assemble(build, data_repo):
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(ROOT / "deploy" / "hf-space", build)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "lunarsolve", build / "lunarsolve", ignore=ignore)
    (build / "server").mkdir()
    for f in ("api.py", "fetch_data.py"):
        shutil.copy(ROOT / "server" / f, build / "server" / f)
    (build / "server" / "__init__.py").write_text("")
    (build / "weights").mkdir()
    shutil.copy(ROOT / "weights" / "craternet.pt", build / "weights" / "craternet.pt")
    (build / "data_repo.txt").write_text(data_repo + "\n")
    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face user (or organization) name")
    ap.add_argument("--space", default="lunar-solver")
    ap.add_argument("--dataset", default="lunarsolve-data")
    ap.add_argument("--skip-data", action="store_true", help="do not upload the data files")
    ap.add_argument("--data-only", action="store_true", help="only upload the data files (no Space)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_repo = f"{args.user}/{args.dataset}"
    space_repo = f"{args.user}/{args.space}"

    missing = [n for n in DATA_FILES if not (DATA_DIR / n).exists()]
    if missing:
        sys.exit(f"missing data files in {DATA_DIR}: {missing} (run scripts/setup_data.py first)")
    build = assemble(ROOT / "deploy" / "build", data_repo)
    size = sum(p.stat().st_size for p in build.rglob("*") if p.is_file()) / 1e6
    print(f"assembled {build} ({size:.0f} MB)")
    print(f"data: {len(DATA_FILES)} files, {sum(DATA_FILES.values())} MB -> dataset {data_repo}")
    print(f"space: {space_repo} -> https://{args.user.lower()}-{args.space.lower()}.hf.space")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    if not args.skip_data:
        api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
        for name in DATA_FILES:
            print("uploading", name, flush=True)
            api.upload_file(path_or_fileobj=str(DATA_DIR / name), path_in_repo=name, repo_id=data_repo,
                            repo_type="dataset", commit_message=f"data: {name}")
    if args.data_only:
        print(f"data uploaded: https://huggingface.co/datasets/{data_repo}")
        return
    api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(build), repo_id=space_repo, repo_type="space",
                      commit_message="deploy lunar plate solver API")
    print(f"done. Build log: https://huggingface.co/spaces/{space_repo}  "
          f"API: https://{args.user.lower()}-{args.space.lower()}.hf.space/api/health")


if __name__ == "__main__":
    main()
