SAM Click-To-Polygon for ArcGIS Pro 🏠➡️⬠
Click a building on your basemap → a local SAM model traces it → real
polygon inserted into your real feature class.
No ArcGIS Pro SDK, no Deep Learning framework install — just a Python toolbox and a tiny localhost service.
*Its not the best but it does work* (:
```
ArcGIS Pro (.pyt, stdlib only)           SAM Sidecar (its own env)
┌───────────────────────────┐  PNG+extent  ┌──────────────────────┐
│ click points (Feature Set)│ ───────────► │ FastAPI :8765        │
│ zoom camera / snapshot    │              │ SAM 2.1 Large (CPU)  │
│ world-file georeferencing │ ◄─────────── │ mask→contour→simplify│
│ InsertCursor into layer   │              |  map-coord rings     │
└───────────────────────────┘              └──────────────────────┘
```

Files:
`SamSegment.pyt`	ArcGIS Pro toolbox (needs nothing installed)
`SamSegment.ClickToPolygon.pyt.xml`	tooltip metadata — keep next to the .pyt
`sam_sidecar.py`	local AI service — runs in its own Python env


Setup:
Sidecar (one time, any Python 3.10+, NOT Pro's env):
```
pip install fastapi uvicorn ultralytics opencv-python-headless pillow numpy
python sam_sidecar.py
```
First use downloads `sam2.1_l.pt` (~900 MB).
or fetch the .pt elsewhere and drop it next to the script.
Pro: Catalog → Toolboxes → Add Toolbox → `SamSegment.pyt`. Done.


Usage:
Start the sidecar, leave the terminal open (python "C:\\sam_sidecar.py")
Open Click To Polygon (SAM)
Click Points → pencil ✏️ → click targets on the map (or drop in a point layer for batch mode)
Pick the target polygon layer, scale, and target type → Run
CPU life: ~15–40 s per click with SAM 2.1 Large ☕


Parameters:
Click Points — one click per target; a point layer works for batch runs
Target Polygon Layer — where polygons are inserted; must be editable.
Optional autofill: add `SAM_Model` (text) / `SAM_Date` (date) fields
Snapshot Scale — how far the tool zooms before its snapshot. Target
should fill ⅓–½ of the frame. House ≈ 1:300, default 1:500, long
commercial ≈ 1:800+, or "Current view scale" to use your exact zoom
Target Type — `Building` grabs the whole structure (won't stop at a
roof ridge); `Small object` grabs just the thing under the click (car,
shed, pool) without lassoing its surroundings
Simplify Tolerance (pixels) — outline smoothing. Low = wiggly +
many vertices, high = clean edges + round corners. Default 2
Hide vector layers during snapshot — parcel lines and labels poison 
what the AI sees; leave checked.


Debugging:
The sidecar writes overlay PNGs to `sam_debug/` (next to wherever it runs):
green = chosen mask, red circle = your click, `_REJECTED` = candidates the
filter vetoed. When results look wrong, these images are the truth.


Notes & gotchas:
Map must use a projected spatial reference (Web Mercator is fine)
The tool drives the map camera during a run — hands off the mouse
SAM traces the roofline, which overhangs walls slightly — same
behavior as every footprint product (MSBFP included)
Model ladder if RAM/speed complains: `sam2.1_b.pt` → `sam2.1_s.pt` →
`mobile_sam.pt` (edit `MODEL_PATH` in the sidecar)

