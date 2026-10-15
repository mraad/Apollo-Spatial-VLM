# Geolocating the point cloud from EXIF

Status: initial idea (written 2026-10-14), updated by the [two-view localization investigation](two-view-localization.md) on 2026-10-15. The newer proposal targets 10 cm at 5–20 m and includes reproducible stereo/uncertainty experiments. Step 1 of the prototype below is done: the demo uses the EXIF 35 mm-equivalent focal length for the camera model (`exif_fov_deg` in `demo.py`). GPS and heading are still ignored.

## Question

If an uploaded photo carries EXIF with GPS position, focal length and similar metadata, can every point (and splat) in the cloud get a geospatial location and a probability?

## Answer

Yes in theory. Each point can get WGS84 latitude, longitude and height plus an uncertainty, best expressed as a per-point covariance (error ellipse) rather than a single probability. Accuracy is limited by camera orientation and by monocular depth error, and EXIF usually lacks part of the orientation.

## What EXIF provides

| Needed | EXIF / metadata source | Typical quality |
|---|---|---|
| Intrinsics (focal length in pixels) | `FocalLength` + `FocalLengthIn35mmFilm` (tag `0xA405`), or sensor size | Good. Replaces the demo's assumed 60° FOV (`DEFAULT_FOV_DEG`) |
| Camera position | GPS IFD (`0x8825`): `GPSLatitude/Longitude/Altitude` | Phone: ~3–10 m horizontal; altitude much worse (10–30 m) |
| Heading | `GPSImgDirection` (written by many phones) | ±5–15°, magnetometer-dependent |
| Pitch and roll | Not in standard EXIF | Phones don't write them. DJI drones write gimbal yaw/pitch/roll and relative altitude in XMP (`drone-dji` namespace) |

Drone photos are the strong case. For phone photos, pitch and roll have to be estimated (horizon or vanishing points, or a ground-plane fit on the depth map) or assumed level.

The real focal length calibrates the depth conversion, but does not eliminate learned depth bias: `DA3METRIC-LARGE` gives metric depth as `focal × output / 300` (`demo.py` uses the 60° guess for `focal`). In the existing evaluation, the Sydney harbour-bridge estimate remained about 18.5 m with either tested focal length; changing focal length alone did not fix it.

## Computation

Points are already in a Three.js-style camera frame (x right, y up, z = −depth). For each point:

1. Rotate by heading, pitch and roll into a local east-north-up (ENU) frame centred on the camera.
2. Add the GPS origin and convert ENU to WGS84 latitude, longitude and ellipsoidal height (`pyproj` or `pymap3d`). Resolve the height datum first: AGL, takeoff-relative, orthometric and ellipsoidal heights are different quantities.

This is one 4×4 transform for the whole cloud. The server can return the transform and the GPS origin, and the browser can geolocate every point and splat (and the VLM targets) without extra payload.

## Uncertainty model

Independent error sources, combined per point:

| Source | Effect | Example |
|---|---|---|
| GPS position | Same offset for every point | σ ≈ 5 m |
| Heading | Lateral error grows with distance: d · σθ | 100 m at 10° → ~17 m |
| Pitch and roll (if estimated or assumed) | Vertical and lateral error grows with distance | Same form as heading |
| Monocular depth | Scene-dependent range bias and local errors | Must be measured on the target domain; no universal percentage guarantee |
| Focal length | Scales depth and lateral offsets | Small with EXIF, large with the 60° guess |
| Per-pixel reliability | Down-weights unreliable pixels | DA3 `Prediction.conf` (`specs.py`); not yet checked whether `DA3METRIC-LARGE` fills it. Depth-edge ("flying") pixels are least reliable |

Rough expectation: within ~20 m of the camera, a few metres; at 100 m and beyond, tens of metres, dominated by heading and depth scale.

A "probability" can be derived from the covariance, e.g. the probability that the true location lies within r metres of the reported one.

## Ways to tighten it

- Register the cloud to known geometry: a DEM/DSM or building footprints (ICP-style). This corrects heading and depth scale together, the two largest unknowns.
- Multiple photos: DA3 any-view models estimate relative poses. A measured baseline establishes scale; two GPS camera centres still leave rotation about the baseline unresolved. Add attitude/gravity or noncollinear control geometry. The bundled DA3 pose-alignment path also needs a two-view adapter; see the newer investigation.
- Drone XMP: gimbal angles remove most of the orientation guesswork.

## Privacy

GPS EXIF is sensitive location data. Decide whether it is kept, logged or returned before reading it. Today the demo drops it after `exif_transpose`.

## Smallest prototype

1. `demo.py`: read EXIF focal length (35 mm equivalent) and GPS; if present, use the real focal length instead of 60°.
2. Return the camera-to-WGS84 transform, the GPS origin and the error-model parameters in `meta`.
3. Show latitude/longitude for the VLM targets in the target list; optionally per point on hover.
4. Test first on a DJI drone photo, which has pitch and roll in XMP.
