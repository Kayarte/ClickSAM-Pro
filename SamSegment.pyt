# -*- coding: utf-8 -*-
"""
SamSegment.pyt — Click imagery in ArcGIS Pro, get a polygon back.
=================================================================
Click a building (or car, shed, pool...) on your basemap. The tool
snapshots the map view, sends it to a local SAM sidecar, and inserts
the returned footprint into the polygon layer of your choice.
Real edits, undo-able, schema-respecting, 100% offline AI.

Requires sam_sidecar.py running on localhost:8765 (see README).
Uses only the Python standard library — nothing to install in Pro's env.

Repo: github.com/Kayarte/ClickSAM-Pro
"""

import base64
import datetime
import json
import os
import tempfile
import urllib.request

import arcpy

SIDECAR = "http://127.0.0.1:8765"
EXPORT_W, EXPORT_H = 1024, 1024


def _post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def _sidecar_alive():
    try:
        with urllib.request.urlopen(SIDECAR + "/health", timeout=3) as r:
            return json.loads(r.read().decode("utf-8")).get("ok", False)
    except Exception:
        return False


class Toolbox(object):
    def __init__(self):
        self.label = "SAM Segment"
        self.alias = "samsegment"
        self.tools = [ClickToPolygon]


class ClickToPolygon(object):
    def __init__(self):
        self.label = "Click To Polygon (SAM)"
        self.description = ("Click imagery, SAM segments the target, and the "
                            "footprint is inserted into your polygon layer.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        pts = arcpy.Parameter(
            displayName="Click Points",
            name="in_points",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Required",
            direction="Input",
        )

        target = arcpy.Parameter(
            displayName="Target Polygon Layer",
            name="target_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        target.filter.list = ["Polygon"]

        scale = arcpy.Parameter(
            displayName="Snapshot Scale",
            name="snap_scale",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        scale.filter.type = "ValueList"
        scale.filter.list = ["Current view scale", "1:150", "1:300", "1:500",
                             "1:800", "1:1200", "1:2000"]
        scale.value = "1:500"

        pick = arcpy.Parameter(
            displayName="Target Type",
            name="mask_pick",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        pick.filter.type = "ValueList"
        pick.filter.list = ["Building", "Small object"]
        pick.value = "Building"

        simplify = arcpy.Parameter(
            displayName="Simplify Tolerance (pixels)",
            name="simplify_px",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
        )
        simplify.value = 2.0

        hide_vec = arcpy.Parameter(
            displayName="Hide vector layers during snapshot",
            name="hide_vectors",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        hide_vec.value = True

        return [pts, target, scale, pick, simplify, hide_vec]

    def execute(self, parameters, messages):
        in_points = parameters[0].value
        target = parameters[1].valueAsText
        scale_txt = parameters[2].valueAsText
        mask_pick = "smallest" if parameters[3].valueAsText == "Small object" else "largest"
        simplify_px = parameters[4].value or 2.0
        hide_vectors = parameters[5].value if parameters[5].value is not None else True

        if not _sidecar_alive():
            arcpy.AddError(f"SAM sidecar not responding at {SIDECAR}.\n"
                           "Start it first:  python sam_sidecar.py")
            return

        aprx = arcpy.mp.ArcGISProject("CURRENT")
        view = aprx.activeView
        if view is None or not hasattr(view, "camera"):
            arcpy.AddError("No active map view. Click into the map, then rerun.")
            return

        sr_map = view.map.spatialReference
        if sr_map.type == "Geographic":
            arcpy.AddWarning("Map uses a geographic SR — switch to a projected "
                             "map (Web Mercator / State Plane) for correct chips.")
        target_sr = arcpy.Describe(target).spatialReference

        pts = []
        with arcpy.da.SearchCursor(in_points, ["SHAPE@"]) as cur:
            for (geom,) in cur:
                g = geom.projectAs(sr_map) if geom.spatialReference.name != sr_map.name else geom
                pts.append((g.centroid.X, g.centroid.Y))
        if not pts:
            arcpy.AddError("No points. Use the pencil to click the map first.")
            return

        use_current = scale_txt.startswith("Current")
        if not use_current:
            scale = float(scale_txt.split(":")[1])
            ground_w = (EXPORT_W / 96.0) * 0.0254 * scale
            ground_h = (EXPORT_H / 96.0) * 0.0254 * scale

        arcpy.AddMessage(f"{len(pts)} point(s) -> SAM ({scale_txt}, {mask_pick} mask)")

        fields = {f.name.lower(): f.name for f in arcpy.ListFields(target)}
        ins_fields = ["SHAPE@"]
        if "sam_model" in fields:
            ins_fields.append(fields["sam_model"])
        if "sam_date" in fields:
            ins_fields.append(fields["sam_date"])

        toggled = []
        if hide_vectors:
            for lyr in view.map.listLayers():
                try:
                    if lyr.visible and (lyr.isFeatureLayer or lyr.isGroupLayer):
                        lyr.visible = False
                        toggled.append(lyr)
                except Exception:
                    pass

        tmpdir = tempfile.mkdtemp(prefix="samchips_")
        made, skipped = 0, 0
        editor = arcpy.da.Editor(arcpy.Describe(target).path)
        editor.startEditing(False, True)
        editor.startOperation()

        try:
            with arcpy.da.InsertCursor(target, ins_fields) as icur:
                for i, (x, y) in enumerate(pts, 1):
                    if use_current:
                        view.camera.X, view.camera.Y = x, y
                    else:
                        view.camera.setExtent(arcpy.Extent(
                            x - ground_w / 2, y - ground_h / 2,
                            x + ground_w / 2, y + ground_h / 2,
                            spatial_reference=sr_map))

                    png = os.path.join(tmpdir, f"chip_{i}.png")
                    view.exportToPNG(png, EXPORT_W, EXPORT_H, world_file=True)

                    # the world file holds the TRUE georef of the export,
                    # which can differ from the camera extent
                    pgw = os.path.splitext(png)[0] + ".pgw"
                    if os.path.exists(pgw):
                        with open(pgw) as f:
                            A, _, _, E, C, F = [float(v) for v in f.read().split()]
                        xmin = C - A / 2.0
                        ymax = F - E / 2.0
                        xmax = xmin + A * EXPORT_W
                        ymin = ymax + E * EXPORT_H
                    else:
                        ext = view.camera.getExtent()
                        xmin, ymin, xmax, ymax = ext.XMin, ext.YMin, ext.XMax, ext.YMax
                        arcpy.AddWarning("  no world file; georef may be off")

                    with open(png, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")

                    resp = _post(SIDECAR + "/segment", {
                        "image_b64": b64,
                        "extent": {"xmin": xmin, "ymin": ymin,
                                   "xmax": xmax, "ymax": ymax},
                        "points": [[x, y]],
                        "simplify_px": simplify_px,
                        "mask_pick": mask_pick,
                        "prompt_spread_px": 10 if mask_pick == "smallest" else 35,
                    })

                    rings = resp["polygons"][0]["rings"]
                    if not rings:
                        why = resp["polygons"][0].get("note", "unknown")
                        arcpy.AddWarning(f"  point {i}: skipped -> {why}")
                        skipped += 1
                        continue

                    poly = arcpy.Polygon(
                        arcpy.Array([
                            arcpy.Array([arcpy.Point(px, py) for px, py in ring])
                            for ring in rings
                        ]),
                        sr_map,
                    ).projectAs(target_sr)

                    row = [poly]
                    if "sam_model" in fields:
                        row.append("sam2.1_l")
                    if "sam_date" in fields:
                        row.append(datetime.datetime.now())
                    icur.insertRow(row)
                    made += 1
                    arcpy.AddMessage(f"  point {i}/{len(pts)}: polygon in "
                                     f"({resp['elapsed_s']}s)")

            editor.stopOperation()
            editor.stopEditing(True)
        except Exception:
            editor.abortOperation()
            editor.stopEditing(False)
            raise
        finally:
            for lyr in toggled:
                try:
                    lyr.visible = True
                except Exception:
                    pass

        arcpy.AddMessage(f"Done. {made} inserted, {skipped} skipped. "
                         "Ctrl+Z if SAM did you dirty.")
