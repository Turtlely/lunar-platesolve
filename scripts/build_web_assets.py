"""Build the static assets of the website (web/assets, web/examples).

    python scripts/build_web_assets.py

moon_4k.jpg   global equirectangular WAC mosaic, 4096 x 2048, longitude -180 at the left edge
              (the texture convention of three.js SphereGeometry)
examples/*    the demo examples as PNG (lossless), with small JPEG thumbnails
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunarsolve.terrain import WacMosaic  # noqa: E402

WEB = ROOT / "web"


def main():
    (WEB / "assets").mkdir(parents=True, exist_ok=True)
    (WEB / "examples").mkdir(parents=True, exist_ok=True)
    wac = WacMosaic()
    W, H = 4096, 2048
    lon = -180 + 360 * (np.arange(W) + 0.5) / W
    lat = 90 - 180 * (np.arange(H) + 0.5) / H
    LON, LAT = np.meshgrid(lon, lat)
    rows = []
    for a in range(0, H, 256):                      # in bands (memory)
        rows.append(wac.sample(LAT[a:a + 256], LON[a:a + 256] % 360, 2 * np.pi * 1737.4 / W))
    img = np.vstack(rows).astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.7])
    img = np.clip((img - lo) / (hi - lo), 0, 1) ** 0.9 * 255
    cv2.imwrite(str(WEB / "assets" / "moon_4k.jpg"), img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    small = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(WEB / "assets" / "moon_1k.jpg"), small.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for p in sorted((ROOT / "demo" / "examples").glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        # lossless: JPEG block artifacts in flat, high-sun images become false crater detections
        cv2.imwrite(str(WEB / "examples" / (p.stem + ".png")), im, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        t = cv2.resize(im, (320, round(320 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(WEB / "examples" / (p.stem + "_thumb.jpg")), t, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("done")


if __name__ == "__main__":
    main()
