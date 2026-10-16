"""Download the data files that the API server needs from a Hugging Face dataset repository.

    python server/fetch_data.py --repo <user>/lunarsolve-data

The files go to LUNARSOLVE_DATA (default ~/.cache/lunarsolve).  The same list is used by
scripts/deploy_hf.py to upload them.  The server does not need the WAC mosaic (6 GB) or the
Clementine albedo; without the mosaic, the "high precision" refinement is off.
"""
import argparse
import os
import shutil
from pathlib import Path

# Same rule as lunarsolve.config, without importing the package (the Docker data layer is built
# before the code is copied, so that code changes do not download the data again).
DATA_DIR = Path(os.environ.get("LUNARSOLVE_DATA", Path.home() / ".cache" / "lunarsolve"))
DEFAULT_REPO = "turtlely/lunarsolve-data"

# path inside the data directory -> approximate size (MB)
DATA_FILES = {
    "robbins_catalog.npz": 31,         # crater catalog (Robbins 2019), compact
    "robbins_catalog_order.npy": 5,    # fixed row order of the catalog (the index stores row numbers)
    "robbins_rim_height.npy": 5,       # LOLA rim heights of the catalog craters
    "quad_index.npz": 421,             # quad index (8.6 M quads)
    "ldem_64.img": 531,                # LOLA LDEM_64, for elevation lookup and terrain
    "ldem_64.lbl": 0,
    "names/moon_names.npz": 2,         # IAU named features
    # CraterNet catalog v2 (scripts/build_catalog_v2.py): second catalog, for small footprints
    "cn_catalog.npz": 72,
    "cn_catalog_order.npy": 10,
    "cn_catalog_rim_height.npy": 10,
    "quad_index_cn.npz": 748,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("LUNARSOLVE_DATA_REPO") or DEFAULT_REPO,
                    help=f"Hugging Face dataset repository (default {DEFAULT_REPO})")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        dst = DATA_DIR / name
        if dst.exists():
            print("have", name)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(args.repo, name, repo_type="dataset")
        shutil.copyfile(p, dst)
        print("got ", name, f"{dst.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
