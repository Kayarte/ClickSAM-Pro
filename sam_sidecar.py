"""
sam_sidecar.py — Local, offline, CPU-friendly SAM segmentation service.
========================================================================
Companion service for SamSegment.pyt (ArcGIS Pro toolbox).
Receives a map snapshot + click points, returns polygon rings in map
coordinates. Runs 100% on localhost — no internet after the model
weights exist on disk.

Setup (any Python 3.10+ env, NOT ArcGIS Pro's):
    pip install fastapi uvicorn ultralytics opencv-python-headless pillow numpy

Model weights auto-download on first use (~900 MB for sam2.1_l).
Air-gapped machine: download once elsewhere, drop the .pt next to this file.
Fallback ladder if RAM complains: sam2.1_b.pt > sam2.1_s.pt > mobile_sam.pt

Run:
    python sam_sidecar.py        ->  http://127.0.0.1:8765

Debug: every request writes overlay PNGs to ./sam_debug/
(green = chosen mask, red circle = your click, _REJECTED = vetoed candidates)

Repo: github.com/<you>/sam-click-to-polygon
"""

import base64
import io
import os
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from ultralytics import SAM

MODEL_PATH = "sam2.1_l.pt"
HOST, PORT = "127.0.0.1", 8765

app = FastAPI(title="SAM Sidecar")
model = None


class Extent(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class SegmentRequest(BaseModel):
    image_b64: str
    extent: Extent
    points: list[list[float]]
    simplify_px: float = 2.0
    min_area_px: int = 200
    debug: bool = True
    prompt_spread_px: int = 35
    mask_pick: str = "largest"  # "largest" (buildings) | "smallest" (cars, sheds)


def get_model():
    global model
    if model is None:
        print(f"Loading {MODEL_PATH} (CPU)...")
        model = SAM(MODEL_PATH)
    return model


def px_to_map(ring_px, ext, w, h):
    xres = (ext.xmax - ext.xmin) / w
    yres = (ext.ymax - ext.ymin) / h
    return [[ext.xmin + float(px) * xres, ext.ymax - float(py) * yres]
            for px, py in ring_px]


def dump_debug(img, mask, pt, path):
    vis = img.copy()
    vis[mask > 0] = (vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), 3)
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH, "loaded": model is not None}


@app.post("/segment")
def segment(req: SegmentRequest):
    t0 = time.time()

    img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    img_area = h * w

    xres = (req.extent.xmax - req.extent.xmin) / w
    yres = (req.extent.ymax - req.extent.ymin) / h
    px_points = [[(x - req.extent.xmin) / xres, (req.extent.ymax - y) / yres]
                 for x, y in req.points]

    m = get_model()
    polygons = []
    if req.debug:
        os.makedirs("sam_debug", exist_ok=True)
    stamp = int(time.time())

    for idx, (pt, map_pt) in enumerate(zip(px_points, req.points)):
        # a lone point on a featureless roof is an ambiguous prompt;
        # a cluster straddling the ridge makes SAM grab the whole building
        d = req.prompt_spread_px
        prompts = [pt] if d <= 0 else [
            pt,
            [pt[0] - d, pt[1]], [pt[0] + d, pt[1]],
            [pt[0], pt[1] - d], [pt[0], pt[1] + d],
        ]
        prompts = [[min(max(px_, 0), w - 1), min(max(py_, 0), h - 1)]
                   for px_, py_ in prompts]

        res = m(arr, points=[prompts], labels=[[1] * len(prompts)], verbose=False)

        if not res or res[0].masks is None or len(res[0].masks.data) == 0:
            polygons.append({"point": map_pt, "rings": [], "note": "no mask from model"})
            continue

        masks = res[0].masks.data.cpu().numpy().astype(np.uint8)
        px, py = int(pt[0]), int(pt[1])

        # keep masks that contain the click and aren't the entire scene
        candidates = []
        for mk in masks:
            if mk.shape != (h, w):
                mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
            a = int(mk.sum())
            if a == 0:
                continue
            edges = (mk[0, :].any() + mk[-1, :].any()
                     + mk[:, 0].any() + mk[:, -1].any())
            if a > 0.88 * img_area or (a > 0.6 * img_area and edges == 4):
                continue
            if mk[min(py, h - 1), min(px, w - 1)] == 0:
                continue
            candidates.append((a, mk))

        if not candidates:
            note = f"{len(masks)} mask(s) but none usable (missed click or scene-sized)"
            polygons.append({"point": map_pt, "rings": [], "note": note})
            if req.debug and len(masks):
                big = max(masks, key=lambda mk: mk.sum())
                if big.shape != (h, w):
                    big = cv2.resize(big, (w, h), interpolation=cv2.INTER_NEAREST)
                dump_debug(arr, big, pt, f"sam_debug/{stamp}_{idx}_REJECTED.png")
            continue

        picker = max if req.mask_pick == "largest" else min
        mask = picker(candidates, key=lambda t: t[0])[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        if req.debug:
            dump_debug(arr, mask, pt, f"sam_debug/{stamp}_{idx}_used.png")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings, dropped = [], 0
        for c in contours:
            if cv2.contourArea(c) < req.min_area_px:
                dropped += 1
                continue
            simp = cv2.approxPolyDP(c, req.simplify_px, closed=True)
            if len(simp) < 3:
                dropped += 1
                continue
            ring = px_to_map(simp.reshape(-1, 2), req.extent, w, h)
            ring.append(ring[0])
            rings.append(ring)

        note = "" if rings else f"mask found but {dropped} contour(s) under min_area_px"
        polygons.append({"point": map_pt, "rings": rings, "note": note})

    return {
        "polygons": polygons,
        "elapsed_s": round(time.time() - t0, 2),
        "image_size": [w, h],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
