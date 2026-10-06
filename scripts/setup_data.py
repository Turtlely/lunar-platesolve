"""Download the public datasets and build the solver's catalog cache and quad index.

    python scripts/setup_data.py            # catalog + LOLA DEM + index (~0.9 GB download)
    python scripts/setup_data.py --wac      # also the 6 GB LROC WAC mosaic (training / benchmarks)

Files go to $LUNARSOLVE_DATA (default ~/.cache/lunarsolve).
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import CATALOG_CSV, DATA_DIR, INDEX_PATH, LDEM_IMG, WAC_TIF  # noqa: E402

ROBBINS_URL = ("https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
               "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018")
LDEM_URL = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/ldem_64.img"
WAC_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"


def fetch(url, dest):
    print(f"downloading {url}\n  -> {dest}")
    tmp = Path(str(dest) + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:8.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wac", action="store_true", help="also download the LROC WAC 100 m mosaic (6 GB)")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CATALOG_CSV.exists():
        zpath = DATA_DIR / "robbins.zip"
        fetch(ROBBINS_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("lunar_crater_database_robbins_2018.csv"))
            with z.open(name) as src, open(CATALOG_CSV, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zpath.unlink()
    if not LDEM_IMG.exists():
        fetch(LDEM_URL, LDEM_IMG)
    if args.wac and not WAC_TIF.exists():
        fetch(WAC_URL, WAC_TIF)
    from lunarsolve.catalog import Catalog
    cat = Catalog()
    print(f"catalog: {len(cat)} craters; computing rim elevations")
    _ = cat.rim_height
    if not INDEX_PATH.exists():
        from lunarsolve.index import build
        build(cat)
    print("done")


if __name__ == "__main__":
    main()
