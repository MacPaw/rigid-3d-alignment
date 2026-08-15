#!/usr/bin/env python3
"""
Evaluate stage-1 model predictions from test_predictions_stage_1_wo_gt.json
against GT meshes from macpaw-research/asset-alignment-pairs-905k.

The JSON uses the VLADataset "action" convention:
  - gt_translations[0]  = action[:3] = -t   (undo translation)
  - gt_rotations[0]     = R.T[:2,:].flatten()  (6-D undo rotation R.T)

GT assembled position  = canonical src mesh (from GLB).
Predicted assembled    = apply GT forward displacement then predicted undo
                         to the canonical src point cloud.

Metrics:
  1. RMSE(T)          – sqrt(mean((pred_t - gt_t)^2)) over 3 components
  2. L2 translation   – Euclidean ||pred_t - gt_t||
  3. Normalised T     – L2 / object diameter
  4. Geodesic R (deg) – angle between GT and predicted undo rotation matrices
  5. ADD-S            – mean nearest-neighbour dist from GT to predicted cloud
  6. ADD-S accuracy   – fraction with ADD-S < threshold * diameter
  7. Chamfer dist     – bidirectional mean Chamfer distance

Optimisations
-------------
- Canonical meshes are loaded once per unique (asset_id, part_id).
- Point transformations use numpy matrix ops; no per-sample mesh copying.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from asset_vla.geometry import geodesic_distance_deg, reconstruct_rotation


MESH_DIR = "/home/disk2/data/meshes"


# ---------------------------------------------------------------------------
# Rotation helpers  (mirrors VLADataset._reconstruct_rotation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def extract_src_mesh(glb_path):
    """
    Load GLB and return the 'src' mesh at its canonical assembled position.
    Mirrors VLADataset._extract_src_tgt.
    """
    scene = trimesh.load_scene(str(glb_path))

    src_mesh = None

    # Strategy 1: geometry dict key "src"
    for key, geom in scene.geometry.items():
        if key.lower() == "src":
            src_mesh = geom.copy()
            break

    # Strategy 2: scene-graph node name containing "src"
    if src_mesh is None:
        for node in scene.graph.nodes:
            if node == "world":
                continue
            if "src" in node.lower():
                try:
                    transform, geom_name = scene.graph[node]
                except Exception:
                    continue
                if geom_name and geom_name in scene.geometry:
                    src_mesh = scene.geometry[geom_name].copy()
                    try:
                        src_mesh.apply_transform(transform)
                    except Exception:
                        pass
                    break

    # Strategy 3: order-based fallback (index 1 = src by VLADataset convention)
    if src_mesh is None:
        geoms = list(scene.geometry.values())
        src_mesh = (geoms[1] if len(geoms) >= 2 else geoms[0]).copy()

    if hasattr(src_mesh, "to_mesh"):
        src_mesh = src_mesh.to_mesh()

    return src_mesh


# ---------------------------------------------------------------------------
# Numpy-based point cloud transformation
# (avoids per-sample mesh copying; mirrors VLADataset displacement logic)
# ---------------------------------------------------------------------------

def _bbox_center(verts):
    """Bounding-box centre of a (N,3) vertex array."""
    return (verts.min(axis=0) + verts.max(axis=0)) / 2.0


def transform_points(pts, verts, gt_trans, R_gt, pred_trans, R_pred_T):
    """
    Apply (GT forward displacement + predicted undo) to a point cloud.

    VLADataset displacement logic:
      1. apply_translation(-action[:3])  =>  translate by -gt_trans
         (action[:3] = gt_trans = -t, so -action[:3] = +t = forward translation)
      2. rotate by R_gt = R_T.T around bbox centre of TRANSLATED mesh
      3. (undo) translate by +pred_trans
      4. (undo) rotate by R_pred_T around bbox centre of (displaced + pred_trans)

    Parameters
    ----------
    pts     : (N,3) sampled surface points (np.float32)
    verts   : (M,3) mesh vertices for exact bbox-centre computation
    gt_trans  : (3,) GT undo translation  = action[:3]
    R_gt      : (3,3) GT forward rotation = reconstruct(gt_rot_6d).T
    pred_trans: (3,) predicted undo translation
    R_pred_T  : (3,3) predicted undo rotation = reconstruct(pred_rot_6d)
    """
    # Step 1: GT forward translation  (-gt_trans on canonical → displaced)
    pts_d   = pts   - gt_trans
    verts_d = verts - gt_trans
    c1 = _bbox_center(verts_d)

    # Step 2: GT forward rotation R_gt around bbox centre of displaced
    pts_r   = (pts_d   - c1) @ R_gt.T + c1
    verts_r = (verts_d - c1) @ R_gt.T + c1

    # Step 3: predicted undo translation
    pts_u   = pts_r   + pred_trans
    verts_u = verts_r + pred_trans
    c3 = _bbox_center(verts_u)

    # Step 4: predicted undo rotation R_pred_T around bbox centre
    pts_f = (pts_u - c3) @ R_pred_T.T + c3

    return pts_f.astype(np.float32)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def adds_metric(gt_pts, pred_pts):
    tree = cKDTree(pred_pts)
    dists, _ = tree.query(gt_pts, k=1)
    return float(dists.mean())


def chamfer_distance(gt_pts, pred_pts):
    tree_pred = cKDTree(pred_pts)
    tree_gt   = cKDTree(gt_pts)
    d_fwd, _ = tree_pred.query(gt_pts,   k=1)
    d_bwd, _ = tree_gt.query(pred_pts,   k=1)
    return float((d_fwd.mean() + d_bwd.mean()) / 2.0)


def object_diameter(pts):
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class MeshCache:
    """Load-once cache for canonical src meshes keyed by (asset_id, part_id)."""

    def __init__(self, mesh_dir, n_pts):
        self.mesh_dir = Path(mesh_dir)
        self.n_pts = n_pts
        self._cache = {}  # (asset_id, part_id) → {"pts": ..., "verts": ...}

    def get(self, asset_id, part_id):
        key = (asset_id, part_id)
        if key not in self._cache:
            glb = self.mesh_dir / str(asset_id) / f"part_{part_id}.glb"
            if not glb.exists():
                self._cache[key] = None
                return None
            try:
                mesh = extract_src_mesh(str(glb))
                pts, _ = trimesh.sample.sample_surface(mesh, self.n_pts)
                self._cache[key] = {
                    "pts":   pts.astype(np.float32),
                    "verts": np.array(mesh.vertices, dtype=np.float64),
                }
            except Exception as exc:
                print(f"  [warn] mesh load failed {asset_id}/part_{part_id}: {exc}")
                self._cache[key] = None
        return self._cache[key]


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(entry, cache, adds_threshold_ratio):
    asset_id = entry["asset_id"]
    part_id  = entry["part_id"]

    gt_trans    = np.asarray(entry["gt_translations"][0],  dtype=np.float64)
    gt_rot_6d   = np.asarray(entry["gt_rotations"][0],     dtype=np.float64)
    pred_trans  = np.asarray(entry["pred_translations"][0], dtype=np.float64)
    pred_rot_6d = np.asarray(entry["pred_rotations"][0],   dtype=np.float64)

    # --- Translation metrics ---
    trans_diff = pred_trans - gt_trans
    trans_l2   = float(np.linalg.norm(trans_diff))
    trans_rmse = float(np.sqrt(np.mean(trans_diff ** 2)))

    # --- Rotation metrics ---
    R_gt_undo   = reconstruct_rotation(gt_rot_6d)    # R.T  (GT undo)
    R_pred_undo = reconstruct_rotation(pred_rot_6d)  # R_pred.T  (pred undo)
    geo_r_deg   = geodesic_distance_deg(R_gt_undo, R_pred_undo)

    # --- Mesh-based metrics ---
    cached = cache.get(asset_id, part_id)
    if cached is None:
        return None

    gt_pts   = cached["pts"]    # canonical src (GT assembled)
    verts    = cached["verts"]
    diam     = object_diameter(gt_pts)
    norm_trans = trans_l2 / diam if diam > 0 else None

    # GT forward rotation  R_gt = R_gt_undo.T  (R.T^T = R)
    R_gt    = R_gt_undo.T
    pred_pts = transform_points(gt_pts, verts, gt_trans, R_gt, pred_trans, R_pred_undo)

    adds_val    = adds_metric(gt_pts, pred_pts)
    cd_val      = chamfer_distance(gt_pts, pred_pts)
    adds_thresh = adds_threshold_ratio * diam
    adds_ok     = adds_val < adds_thresh

    return {
        "asset_id":       asset_id,
        "part_id":        part_id,
        "trans_rmse":     trans_rmse,
        "trans_l2":       trans_l2,
        "norm_trans":     norm_trans,
        "geo_r_deg":      geo_r_deg,
        "diameter":       diam,
        "adds":           adds_val,
        "adds_threshold": adds_thresh,
        "adds_correct":   adds_ok,
        "chamfer":        cd_val,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _stats(vals):
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(np.median(a))


def _row(label, mean, std, median, unit="", W=46):
    print(
        f"  {label:<{W}} mean={mean:.4f} ± {std:.4f}"
        f"  median={median:.4f}"
        + (f"  [{unit}]" if unit else "")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stage-1 predictions against VLA GT meshes"
    )
    parser.add_argument("--predictions",    default="test_predictions_stage_1_wo_gt.json")
    parser.add_argument("--mesh_dir",       default=MESH_DIR)
    parser.add_argument("--n_pts",          type=int,   default=10_000)
    parser.add_argument("--adds_threshold", type=float, default=0.1)
    parser.add_argument("--output_json",    default=None)
    args = parser.parse_args()

    with open(args.predictions) as fh:
        predictions = json.load(fh)

    print(f"Loading canonical meshes for {len(predictions)} samples …")
    cache = MeshCache(args.mesh_dir, args.n_pts)

    # Pre-warm cache
    unique_keys = {(e["asset_id"], e["part_id"]) for e in predictions}
    for i, (aid, pid) in enumerate(sorted(unique_keys)):
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(unique_keys)} meshes …")
        cache.get(aid, pid)
    print(f"  loaded {len(unique_keys)} unique meshes.\n")

    all_results = []
    for entry in predictions:
        r = evaluate_sample(entry, cache, args.adds_threshold)
        if r is not None:
            all_results.append(r)

    if not all_results:
        sys.exit("[error] no valid results found.")

    n = len(all_results)
    W = 46

    rmse_t    = [r["trans_rmse"]   for r in all_results]
    l2_t      = [r["trans_l2"]     for r in all_results]
    norm_t    = [r["norm_trans"]   for r in all_results if r["norm_trans"] is not None]
    geo_r     = [r["geo_r_deg"]    for r in all_results]
    adds_vals = [r["adds"]         for r in all_results]
    adds_ok   = [r["adds_correct"] for r in all_results]
    cd_vals   = [r["chamfer"]      for r in all_results]

    sep = "=" * 74
    print(f"{sep}")
    print(f"  Evaluation summary  ({n} / {len(predictions)} samples)")
    print(sep)

    _row("RMSE(T)  [= L2/√3, raw]",          *_stats(rmse_t),   W=W)
    _row("L2 translation error",               *_stats(l2_t),    W=W)
    if norm_t:
        _row("Normalised translation error",   *_stats(norm_t),  W=W)
    _row("Geodesic rotation error",            *_stats(geo_r),   unit="deg", W=W)
    _row("ADD-S",                              *_stats(adds_vals), W=W)

    pct   = args.adds_threshold * 100
    n_ok  = sum(adds_ok)
    acc   = 100.0 * n_ok / len(adds_ok)
    label = f"ADD-S accuracy (< {pct:.0f}% diam)"
    print(f"  {label:<{W}} {acc:.2f}%  ({n_ok}/{len(adds_ok)})")

    _row("Chamfer distance", *_stats(cd_vals), W=W)
    print(sep)

    if args.output_json:
        out = {
            "n_samples": n,
            "adds_threshold_ratio": args.adds_threshold,
            "aggregate": {
                "rmse_t_mean":        float(np.mean(rmse_t)),
                "rmse_t_std":         float(np.std(rmse_t)),
                "rmse_t_median":      float(np.median(rmse_t)),
                "l2_t_mean":          float(np.mean(l2_t)),
                "l2_t_median":        float(np.median(l2_t)),
                "norm_trans_mean":    float(np.mean(norm_t))  if norm_t else None,
                "norm_trans_median":  float(np.median(norm_t)) if norm_t else None,
                "geo_r_deg_mean":     float(np.mean(geo_r)),
                "geo_r_deg_std":      float(np.std(geo_r)),
                "geo_r_deg_median":   float(np.median(geo_r)),
                "adds_mean":          float(np.mean(adds_vals)),
                "adds_median":        float(np.median(adds_vals)),
                "adds_accuracy":      acc / 100.0,
                "chamfer_mean":       float(np.mean(cd_vals)),
                "chamfer_median":     float(np.median(cd_vals)),
            },
            "per_sample": all_results,
        }
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nPer-sample results written to: {args.output_json}")


if __name__ == "__main__":
    main()
