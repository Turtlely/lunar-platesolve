---
title: Lunar Plate Solver
emoji: 🌕
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Find where on the Moon a photo was taken, from the craters.
---

# Lunar Plate Solver API

Give one image of the lunar surface. The API finds the craters, matches their pattern against
1.3 million catalogued craters, and returns the camera latitude, longitude and altitude.

```bash
curl -F image=@photo.png https://<this-space>.hf.space/api/solve
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /api/health` | | status, version |
| `POST /api/solve` | multipart `image`; optional `fov_deg`, `parity`, `max_time`, `labels` | camera position and 1-sigma, camera model, footprint, detections, named features |
| `GET /api/lookup?lat=&lon=` | ground point | LOLA elevation, nearest named feature |

Interactive docs: `/docs`. Source and method: see the project repository.
