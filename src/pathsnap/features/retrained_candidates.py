"""28-feature candidate generation adapted from RealAdhi/ML2026.

The trained Random Forest expects these 28 columns in this exact order. ``feature_trip``
converts PathSnap's validated input to the model's local projected-metre representation.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree

from pathsnap.data.reader import Trip as PathSnapTrip


def feature_trip(trip: PathSnapTrip):
    """Adapt a validated PathSnap trip to the teammate model's projected-trip interface."""
    good = np.isfinite(trip.gps[:, :2]).all(axis=1)
    lon0 = float(np.median(trip.gps[good, 0]))
    lat0 = float(np.median(trip.gps[good, 1]))
    crs = (f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 "
           "+units=m +no_defs")
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    px, py = forward.transform(trip.gps[:, 0], trip.gps[:, 1])
    nx, ny = forward.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    return SimpleNamespace(
        trip_id=trip.trip_id, lon=trip.gps[:, 0], lat=trip.gps[:, 1], t=trip.gps[:, 2],
        px=np.asarray(px), py=np.asarray(py), node_x=np.asarray(nx), node_y=np.asarray(ny),
        node_lon=trip.nodes[:, 0], node_lat=trip.nodes[:, 1], arcs=trip.arcs,
        route=trip.route if trip.route.size else None, inv=inverse,
    )

RADIUS_M = 60.0      # candidate search radius
MAX_CANDS = 12       # nearest candidates kept per observation
GAP_S = 5.0          # dt above this is treated as a GPS gap for kinematic features

FEATURES = [
    "d", "d_rank", "d_rel", "d_ratio", "frac", "arc_len", "d_to_start", "d_to_end",
    "cos_h", "abs_ang", "cos_h_prev", "cos_h_next", "speed", "step_prev", "step_next",
    "dt_prev", "dt_next", "deg_from", "deg_to", "outdeg_to", "has_reverse", "n_cands",
    "n_dir_ok", "d_prev", "d_next", "dprog_ok", "min_d_dir_ok", "d_vs_best_dir",
]


def arc_xy(trip: Any):
    a, b = trip.arcs[:, 0], trip.arcs[:, 1]
    return trip.node_x[a], trip.node_y[a], trip.node_x[b], trip.node_y[b]


def seg_dist(px, py, ax, ay, bx, by):
    """Distance from points to segments and the clamped projection fraction."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def _unit(dx, dy):
    m = np.hypot(dx, dy)
    ok = m > 1e-6
    safe = np.where(ok, m, 1.0)
    return np.where(ok, dx / safe, np.nan), np.where(ok, dy / safe, np.nan)


def gps_kinematics(trip: Any):
    """Heading (unit vector), speed and step lengths per observation; NaN across gaps."""
    n = len(trip.t)
    px, py, t = trip.px, trip.py, trip.t
    dt_prev = np.full(n, np.nan); dt_next = np.full(n, np.nan)
    dt_prev[1:] = np.diff(t); dt_next[:-1] = np.diff(t)
    sp = np.full(n, np.nan); sn = np.full(n, np.nan)
    sp[1:] = np.hypot(np.diff(px), np.diff(py)); sn[:-1] = sp[1:]

    pdx = np.full(n, np.nan); pdy = np.full(n, np.nan)
    ndx = np.full(n, np.nan); ndy = np.full(n, np.nan)
    pdx[1:], pdy[1:] = np.diff(px), np.diff(py)
    ndx[:-1], ndy[:-1] = np.diff(px), np.diff(py)
    pdx = np.where(dt_prev <= GAP_S, pdx, np.nan); pdy = np.where(dt_prev <= GAP_S, pdy, np.nan)
    ndx = np.where(dt_next <= GAP_S, ndx, np.nan); ndy = np.where(dt_next <= GAP_S, ndy, np.nan)
    hpx, hpy = _unit(pdx, pdy)
    hnx, hny = _unit(ndx, ndy)
    both_nan = np.isnan(pdx) & np.isnan(ndx)
    cdx = np.where(both_nan, np.nan, np.nan_to_num(pdx) + np.nan_to_num(ndx))
    cdy = np.where(both_nan, np.nan, np.nan_to_num(pdy) + np.nan_to_num(ndy))
    hcx, hcy = _unit(cdx, cdy)
    sp_safe = np.where(dt_prev > 0, dt_prev, 1.0)
    sn_safe = np.where(dt_next > 0, dt_next, 1.0)
    speed = np.where(dt_next <= GAP_S, sn / sn_safe, sp / sp_safe)
    return dict(hcx=hcx, hcy=hcy, hpx=hpx, hpy=hpy, hnx=hnx, hny=hny, speed=speed,
                step_prev=sp, step_next=sn, dt_prev=dt_prev, dt_next=dt_next)


def reference_alignment(trip: Any, k: int = 6, back_penalty: float = 1e3, fwd_pen: float = 0.02):
    """Assign every GPS observation to a reference-route arc.

    The .route file is an ordered arc list with no per-observation alignment. We solve a monotone
    dynamic programme over route positions (cost = point-to-arc distance, backward moves heavily
    penalised) giving ref_pos (index into the route) and ref_arc (arc id) for each observation.
    """
    rt = trip.route
    ax, ay, bx, by = (v[rt] for v in arc_xy(trip))
    n, m = len(trip.px), len(rt)
    kk = min(k, m)
    cand = np.empty((n, kk), dtype=np.int64); dist = np.empty((n, kk))
    CH = max(1, int(4_000_000 // max(m, 1)))
    for s in range(0, n, CH):
        e = min(n, s + CH)
        D, _ = seg_dist(trip.px[s:e, None], trip.py[s:e, None], ax[None], ay[None], bx[None], by[None])
        idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        cand[s:e] = idx; dist[s:e] = np.take_along_axis(D, idx, axis=1)
    cost = dist[0].copy(); back = np.zeros((n, kk), dtype=np.int64)
    for i in range(1, n):
        dj = cand[i][None, :] - cand[i - 1][:, None]
        trans = np.where(dj >= 0, fwd_pen * np.minimum(dj, 50), back_penalty + (-dj) * 0.1)
        tot = cost[:, None] + trans
        bi = tot.argmin(axis=0)
        cost = tot[bi, np.arange(kk)] + dist[i]
        back[i] = bi
    j = int(cost.argmin()); pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = cand[i, j]; j = back[i, j]
    return pos, rt[pos]


def generate(trip: Any, with_labels: bool = True, radius: float = RADIUS_M, max_cands: int = MAX_CANDS):
    """Return (DataFrame of candidate rows with features, per-observation info dict)."""
    ax, ay, bx, by = arc_xy(trip)
    n = len(trip.px)
    geoms = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    tree = STRtree(geoms)
    ok = np.isfinite(trip.px) & np.isfinite(trip.py)
    pts = shapely.points(trip.px, trip.py)
    qi, ai = tree.query(pts[ok], predicate="dwithin", distance=radius)
    oi = np.flatnonzero(ok)[qi]
    have = np.zeros(n, bool); have[oi] = True
    miss = np.flatnonzero(ok & ~have)
    if len(miss):  # nothing within the radius -> keep the single nearest arc
        near = tree.query_nearest(pts[miss], all_matches=False)
        oi = np.concatenate([oi, miss[near[0]]]); ai = np.concatenate([ai, near[1]])
    d, frac = seg_dist(trip.px[oi], trip.py[oi], ax[ai], ay[ai], bx[ai], by[ai])
    df = pd.DataFrame({"obs": oi, "arc": ai, "d": d, "frac": frac})
    df = df.sort_values(["obs", "d", "arc"], kind="stable")
    df["d_rank"] = df.groupby("obs").cumcount()
    df = df[df.d_rank < max_cands].reset_index(drop=True)
    o, a = df["obs"].values, df["arc"].values
    kin = gps_kinematics(trip)

    alen = np.hypot(bx - ax, by - ay)[a]
    df["arc_len"] = alen
    df["d_to_start"] = df["frac"] * alen
    df["d_to_end"] = (1 - df["frac"]) * alen
    abx, aby = (bx - ax)[a], (by - ay)[a]
    abm = np.hypot(abx, aby); abm = np.where(abm > 0, abm, 1.0)
    ux, uy = abx / abm, aby / abm

    def cosv(hx, hy):
        return hx[o] * ux + hy[o] * uy
    df["cos_h"] = cosv(kin["hcx"], kin["hcy"])
    df["cos_h_prev"] = cosv(kin["hpx"], kin["hpy"])
    df["cos_h_next"] = cosv(kin["hnx"], kin["hny"])
    df["abs_ang"] = np.degrees(np.arccos(np.clip(df["cos_h"], -1, 1)))
    for key in ("speed", "step_prev", "step_next", "dt_prev", "dt_next"):
        df[key] = kin[key][o]
    n_nodes = len(trip.node_x)
    outdeg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
    indeg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
    df["deg_from"] = (outdeg + indeg)[trip.arcs[a, 0]]
    df["deg_to"] = (outdeg + indeg)[trip.arcs[a, 1]]
    df["outdeg_to"] = outdeg[trip.arcs[a, 1]]
    keys = trip.arcs[:, 0].astype(np.int64) * n_nodes + trip.arcs[:, 1]
    rkeys = trip.arcs[a, 1].astype(np.int64) * n_nodes + trip.arcs[a, 0]
    df["has_reverse"] = np.isin(rkeys, keys).astype(np.int8)

    g = df.groupby("obs")
    dmin = g["d"].transform("min")
    df["d_rel"] = df["d"] - dmin
    df["d_ratio"] = df["d"] / (dmin + 1.0)
    df["n_cands"] = g["d"].transform("size")
    dir_ok = df["cos_h"] > 0.5
    df["_dok"] = dir_ok.astype(np.int8)
    df["n_dir_ok"] = df.groupby("obs")["_dok"].transform("sum")
    dd = df["d"].where(dir_ok)
    df["min_d_dir_ok"] = dd.groupby(df["obs"]).transform("min")
    df["d_vs_best_dir"] = df["d"] - df["min_d_dir_ok"]
    df.drop(columns="_dok", inplace=True)

    # neighbour consistency: distance of the previous / next fix to this same candidate arc
    tprog = None
    for name, sh in (("d_prev", -1), ("d_next", 1)):
        no = np.clip(o + sh, 0, n - 1)
        dn, tn = seg_dist(trip.px[no], trip.py[no], ax[a], ay[a], bx[a], by[a])
        valid = (o + sh >= 0) & (o + sh < n)
        df[name] = np.where(valid, dn, np.nan)
        if sh == 1:
            tprog = tn
    df["dprog_ok"] = np.where(np.isnan(df["d_next"]), np.nan,
                              ((tprog - df["frac"]) * alen) / (df["step_next"] + 1.0))

    info = dict(n=n)
    if with_labels and trip.route is not None:
        pos, ref = reference_alignment(trip)
        df["label"] = (df["arc"].values == ref[o]).astype(np.int8)
        info.update(ref_pos=pos, ref_arc=ref)
    df.insert(0, "trip", trip.trip_id)
    return df, info
