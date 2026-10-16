"""Solve with several crater catalogs in turn, each read by the detector that suits it.

    from lunarsolve.multi import default_solver
    s = default_solver()                # Robbins first, then the CraterNet catalog (if installed)
    sol = s.solve(img=img, max_time=30)
    sol.catalog_name                    # which catalog solved it

Why two catalogs.  Robbins (1.3 M craters) solves large footprints best: its index is smaller and
complete above 1-2 km.  The CraterNet catalog (scripts/build_catalog_v2.py: the detector itself,
several scales and lightings, merged with Robbins) adds craters down to 0.5 km and builds its index
only from craters the detector sees; it solves small footprints far better (Apollo crops 23 km
wide: 16/36 against 3/36).  The first catalog gets `shares[0]` of the time budget, the next the
rest of it, and so on.
"""
import time

from .solve import Solver


class MultiSolver:
    def __init__(self, solvers, names, shares=None):
        self.solvers = solvers
        self.names = names
        self.shares = shares or [1.0 / len(solvers)] * len(solvers)
        self.stats = {}
        self.load_time = sum(getattr(x, "load_time", 0.0) for x in solvers)

    # the first solver stands in for attributes that callers read (detector, catalog of record)
    @property
    def model(self):
        return self.solvers[0].model

    @property
    def cat(self):
        return self.solvers[0].cat

    @property
    def index(self):
        return self.solvers[0].index

    def detect(self, img):
        return self.solvers[0].detect(img)

    def solve(self, img=None, max_time=20.0, **kw):
        t0 = time.time()
        for k, (s, name) in enumerate(zip(self.solvers, self.names)):
            left = max_time - (time.time() - t0)
            if left < 2:
                break
            share = self.shares[k] / sum(self.shares[k:])          # of what is left
            budget = left if k == len(self.solvers) - 1 else max(left * share, 2.0)
            sol = s.solve(img=img, max_time=budget, **kw)
            self.stats = dict(s.stats, catalog=name)
            if sol is not None:
                sol.catalog_name = name
                sol.catalog = s.cat                     # matches refer to this catalog's rows
                return sol
        return None


def default_solver(device=None, shares=(0.5, 0.5)):
    """Robbins, then the CraterNet catalog when its files exist; else Robbins alone."""
    from .catalog import Catalog
    from .config import CN_CATALOG_NPZ, CN_INDEX_PATH, WEIGHTS_V3_PATH
    from .detect import load_model
    from .index import QuadIndex
    first = Solver(device=device)                      # Robbins with the default detector
    if not (CN_CATALOG_NPZ.exists() and CN_INDEX_PATH.exists() and WEIGHTS_V3_PATH.exists()):
        return MultiSolver([first], ["Robbins"])
    # the CraterNet catalog was built by the v3 detector: read images with the same detector
    second = Solver(catalog=Catalog(CN_CATALOG_NPZ), index=QuadIndex(CN_INDEX_PATH),
                    model=load_model(WEIGHTS_V3_PATH, device=first.device), device=first.device)
    return MultiSolver([first, second], ["Robbins", "CraterNet"], list(shares))
