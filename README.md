# lunar-platesolve

Blind plate solving for the Moon. Give it one image of the lunar surface and nothing
else. It finds where on the Moon the image is, and where the camera was: latitude,
longitude and altitude, plus pointing, heading and the image's ground footprint.

The idea is the same as star plate solving (astrometry.net), with impact craters in place
of stars:

| stars | craters |
|---|---|
| star centroid | crater centre |
| brightness (ranks stars) | diameter (ranks craters, **and** is a measured invariant) |
| star catalog (Gaia, Tycho) | Robbins 2018 lunar crater database, 1.3 M craters ≥ 1 km |
| quad of 4 stars → 4-D hash code | quad of 4 craters → 8-D hash code (geometry + radius ratios) |
| sky is a sphere seen from inside | Moon is a sphere seen from outside → full pinhole camera fit |

## How it works

```
image ──► CraterNet ──► craters (x, y, r) ──► quads ──► KD-tree lookup ──► hypotheses
                                                                 │
          camera fit ◄── grow & verify (log-odds) ◄──────────────┘
          (lat, lon, alt, attitude, focal length)
```

### 1. Crater detection: `lunarsolve/model.py`, `lunarsolve/synth.py`

A crater looks different under every sun angle: a bright crescent at low sun, a dark ring
at high sun, half of it in black shadow. Classical circle fitting fails on this. **CraterNet**
is a 6 M-parameter U-Net with CenterNet-style heads (stride 2). It predicts a heatmap of
crater *centres* and regresses the radius. It learns where the centre is from the shading
pattern, whatever the lighting.

Training data comes from two sources, both labelled from the Robbins catalog:

* **Synthetic renders of the LOLA DEM** (GPU). A random location, scale (0.6 to 4 km/px),
  sun elevation (1.5° to 75°) and azimuth. Lunar-Lambert shading, cast shadows (a running
  maximum along sun-aligned rows), sphere curvature, albedo noise, blur, gamma and sensor
  noise. Mild anisotropy simulates oblique views. This covers lighting the real mosaic
  does not have.
* **Real LROC WAC 100 m mosaic crops** (CPU workers, 0.25 to 3 km/px). These cover real
  texture and albedo.

Craters outside the trainable radius band (3 to 48 px), or below the catalog's completeness
limit, are *ignored* in the loss, not treated as negatives. At inference, an image pyramid
extends the radius range. Longitudes 330° to 360° are held out of training.

### 2. Quad index: `lunarsolve/quads.py`, `lunarsolve/index.py`

For four craters A, B, C, D (A–B the most distant pair, C and D inside the circle with
diameter AB), place A at (0, 0) and B at (1, 0). The code is

```
[xC, yC, xD, yD,  wR·(ln rA − m), wR·(ln rB − m), wR·(ln rC − m),  wS·(m − ln|AB|)]
```

Here m is the mean log radius. All eight numbers stay the same under any rotation, scale and
translation of the image. Symmetry is broken as in astrometry.net, and queries near a
symmetry boundary try the alternative labellings. The radius weights *whiten* the code: they
scale log-radius noise (about 0.08) to positional noise (about 0.02). This turned out to be
the most important tuning choice. It cut false index hits by about 30 times at the same recall.

The index has 15 scale levels (quad size 8 km to 1450 km). At each level, a Fibonacci lattice
covers the sphere, and at each lattice point the 9 largest nearby craters of a suitable size
form up to 6 quads. That is 8.6 M quads in one 8-D KD-tree.

### 3. Solver: `lunarsolve/solve.py`

1. Query quads come from two sources, interleaved: the N largest detections
   (N = 8 → 40, like brightest stars), and a *local rule* that reproduces the index
   construction in pixel space. The local rule matters in crowded terrain, where the
   globally largest craters do not make the locally chosen index quads.
2. Index hits are verified **closest-code-first**. Chance hits fill the 8-D tolerance ball
   uniformly, so they pile up near its edge, while true hits sit near the centre.
3. Each hit gives a similarity transform to a local tangent plane. A cheap check against the
   catalog filters it. Survivors grow outward from the quad, from similarity to affine to
   homography. They are then scored with a log-likelihood ratio: matched detections against
   the chance of a random catalog crater of similar size nearby. Detections below the
   catalog's completeness limit carry no weight either way.
4. A full pinhole camera is fitted (robust PnP RANSAC init, then robust least squares). All
   detections are re-matched through the camera and the fit is repeated. Without a known
   focal length, the fit is profiled over field of view, then the focal length is freed.
   The final camera solution must pass the log-odds threshold again.

## Results

RESULTS_PLACEHOLDER

## Usage

```bash
python -m lunarsolve image.png
python -m lunarsolve image.png --fov 20 --overlay solved.png --json result.json
```

```python
from lunarsolve.solve import Solver
solver = Solver()                      # loads catalog + index (a few seconds), reuse it
sol = solver.solve(img=gray_float_image, fov_deg=None)
print(sol.summary())
lat, lon = sol.pixel_to_latlon(u, v)
```

Output fields: camera latitude, longitude and altitude (km above the 1737.4 km sphere), its
uncertainty, focal length / FOV (given or estimated), image-centre latitude and longitude,
GSD, heading of image-up (clockwise from north), off-nadir angle, mirror flag, matched
craters, RMS residual and log-odds.

### About altitude

The image scale fixes only *altitude / focal length*. The altitude itself comes from
perspective: keystone distortion from tilt, and the curvature of the Moon across the
footprint. For wide or oblique views this is enough, and the solver estimates both. For
narrow near-nadir views it is not. The solver then reports a large `altitude_sigma_km`,
and you should pass `--fov` or `--focal`.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
python scripts/setup_data.py          # Robbins catalog + LOLA DEM (~0.8 GB), builds the index
python scripts/setup_data.py --wac    # + LROC WAC mosaic (6 GB), only for training / benchmarks
```

Data goes to `$LUNARSOLVE_DATA` (default `~/.cache/lunarsolve`). The trained detector
weights are in `weights/craternet.pt`.

### Training and benchmarks

```bash
python -m lunarsolve.train --iters 25000          # ~1.7 h on an RTX 4060 laptop GPU
python scripts/eval_sim.py --mode fake            # solver only: catalog craters + noise
python scripts/eval_sim.py --mode wac             # real WAC imagery through random cameras
python scripts/eval_sim.py --mode dem             # LOLA renders with random sun angles
python scripts/eval_sim.py --mode wac --lon 330 360   # held-out longitudes only
python -m pytest tests
```

## Limitations

LIMITATIONS_PLACEHOLDER

## Data sources

* Robbins, S. J. (2019), *A new global database of lunar impact craters > 1–2 km*,
  JGR Planets 124, 871–892. Distributed by USGS Astrogeology.
* LRO LOLA gridded elevation model LDEM_64 (NASA PDS Geosciences Node).
* LRO LROC WAC global morphologic mosaic, 100 m/px (NASA / GSFC / Arizona State University,
  distributed by USGS).
