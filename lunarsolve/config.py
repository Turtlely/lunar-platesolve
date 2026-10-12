"""Paths and physical constants."""
import os
from pathlib import Path

MOON_RADIUS_KM = 1737.4

# Large data (DEM, mosaic, catalog, index) lives outside the repo.
DATA_DIR = Path(os.environ.get("LUNARSOLVE_DATA", Path.home() / ".cache" / "lunarsolve"))

REPO_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_DIR / "weights" / "craternet.pt"

CATALOG_CSV = DATA_DIR / "lunar_crater_database_robbins_2018.csv"
CATALOG_NPZ = DATA_DIR / "robbins_catalog.npz"
LDEM_IMG = DATA_DIR / "ldem_64.img"          # LOLA 64 px/deg int16, 0.5 m DN
LDEM_PPD = 64
WAC_TIF = DATA_DIR / "wac_100m.tif"          # LROC WAC global mosaic, 8-bit
INDEX_PATH = DATA_DIR / "quad_index.npz"
# Fine catalog: Robbins + CraterNet detections on the WAC 100 m mosaic (D >= ~0.65 km)
FINE_CATALOG_NPZ = DATA_DIR / "fine_catalog.npz"
FINE_INDEX_PATH = DATA_DIR / "quad_index_fine.npz"
