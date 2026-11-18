"""HMM / Viterbi route reconstruction adapted from RealAdhi/ML2026."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd



MAX_PRECOMPUTE_GAP_S = 60.0   # larger time gaps never use network distances


@dataclass
class DecoderParams:
    w_emit: float = 1.0       # weight of classifier log-probability
    w_dist: float = 0.0       # weight of the Gaussian GPS-distance emission term
    sigma: float = 15.0       # metres, for the distance term
    beta: float = 30.0        # metres, transition scale
    w_trans: float = 1.0      # weight of the transition term
    break_pen: float = 8.0    # log-penalty for unreachable transitions
    uturn_pen: float = 2.0    # extra log-penalty for immediately reversing onto the opposite arc
    gap_s: float = 30.0       # dt above this => transition ignored (restart)
    eps: float = 1e-4

    def as_dict(self):
        return self.__dict__.copy()


def _adjacency(trip, alen):
    n_nodes = len(trip.node_x)
    order = np.argsort(trip.arcs[:, 0], kind="stable")
    starts = np.searchsorted(trip.arcs[order, 0], np.arange(n_nodes + 1))
    return order, starts


def _dijkstra(order, starts, arcs, alen, src, targets, limit):
    """Shortest distances (m) from node src to the target nodes, bounded by limit."""
    dist = {src: 0.0}
    out = {}
    remaining = set(targets)
    if src in remaining:
        out[src] = 0.0; remaining.discard(src)
    heap = [(0.0, src)]
    done = set()
    while heap and remaining:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if d > limit:
            break
        if u in remaining:
            out[u] = d; remaining.discard(u)
            if not remaining:
                break
        for k in range(starts[u], starts[u + 1]):
            a = order[k]
            v = arcs[a, 1]
            nd = d + alen[a]
            if nd < dist.get(v, 1e18) and nd <= limit:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return out


@dataclass
class Precomputed:
    obs: np.ndarray            # observation index of each step
    row_start: np.ndarray      # candidate-row offsets (len steps+1) into the candidate frame
    rd: list                   # per transition: (K_t, K_{t+1}) route distance, inf if unreachable (or None)
    gc: np.ndarray             # straight-line step distance between consecutive steps
    dt: np.ndarray
    uturn: list                # per transition: bool matrix
    arc: np.ndarray            # candidate arc per row
    d: np.ndarray              # GPS->arc distance per row


def precompute(trip, df: pd.DataFrame) -> Precomputed:
    df = df.reset_index(drop=True)
    ax, ay = trip.node_x[trip.arcs[:, 0]], trip.node_y[trip.arcs[:, 0]]
    bx, by = trip.node_x[trip.arcs[:, 1]], trip.node_y[trip.arcs[:, 1]]
    alen = np.hypot(bx - ax, by - ay)
    order, starts = _adjacency(trip, alen)
    obs = df["obs"].values
    uo, first = np.unique(obs, return_index=True)
    row_start = np.append(first, len(df))
    arc = df["arc"].values; frac = df["frac"].values
    rd, ut = [], []
    gc = np.zeros(max(len(uo) - 1, 0)); dtv = np.zeros_like(gc)
    for t in range(len(uo) - 1):
        i, j = uo[t], uo[t + 1]
        gc[t] = np.hypot(trip.px[j] - trip.px[i], trip.py[j] - trip.py[i])
        dtv[t] = trip.t[j] - trip.t[i]
        sl0, sl1 = slice(row_start[t], row_start[t + 1]), slice(row_start[t + 1], row_start[t + 2])
        a0, a1 = arc[sl0], arc[sl1]
        f0, f1 = frac[sl0], frac[sl1]
        rev = (trip.arcs[a0, 0][:, None] == trip.arcs[a1, 1][None]) & (trip.arcs[a0, 1][:, None] == trip.arcs[a1, 0][None])
        ut.append(rev & (a0[:, None] != a1[None]))
        if dtv[t] > MAX_PRECOMPUTE_GAP_S:
            rd.append(None); continue
        limit = min(3.0 * gc[t] + 150.0, 3000.0)
        heads = np.unique(trip.arcs[a0, 1]); tails = np.unique(trip.arcs[a1, 0])
        sp = {int(h): _dijkstra(order, starts, trip.arcs, alen, int(h), tails.tolist(), limit) for h in heads}
        M = np.full((len(a0), len(a1)), np.inf)
        for r in range(len(a0)):
            h = int(trip.arcs[a0[r], 1]); spd = sp[h]
            for c in range(len(a1)):
                if a0[r] == a1[c]:
                    M[r, c] = abs(f1[c] - f0[r]) * alen[a0[r]]
                else:
                    s = spd.get(int(trip.arcs[a1[c], 0]))
                    if s is not None:
                        M[r, c] = (1 - f0[r]) * alen[a0[r]] + s + f1[c] * alen[a1[c]]
        rd.append(M)
    return Precomputed(uo, row_start, rd, gc, dtv, ut, arc, df["d"].values)


def viterbi(pre: Precomputed, p: np.ndarray, prm: DecoderParams) -> np.ndarray:
    """Return the decoded arc id for each observation step (len = number of observations)."""
    T = len(pre.obs)
    rs = pre.row_start
    sl = lambda t: slice(rs[t], rs[t + 1])
    # emission log-scores
    em = np.empty(len(p))
    for t in range(T):
        s = sl(t)
        q = np.clip(p[s], prm.eps, None); q = q / q.sum()
        em[s] = prm.w_emit * np.log(q) - prm.w_dist * (pre.d[s] ** 2) / (2 * prm.sigma ** 2)
    back = []
    delta = em[sl(0)].copy()
    for t in range(1, T):
        k0, k1 = rs[t] - rs[t - 1], rs[t + 1] - rs[t]
        M = pre.rd[t - 1]
        if M is None or pre.dt[t - 1] > prm.gap_s:
            tr = np.zeros((k0, k1))
        else:
            tr = -np.abs(pre.gc[t - 1] - M) / prm.beta
            tr = np.where(np.isfinite(M), tr, -prm.break_pen / max(prm.w_trans, 1e-9))
            tr = tr - prm.uturn_pen / max(prm.w_trans, 1e-9) * pre.uturn[t - 1]
            tr = tr * prm.w_trans
        tot = delta[:, None] + tr
        bi = tot.argmax(axis=0)
        delta = tot[bi, np.arange(k1)] + em[sl(t)]
        back.append(bi)
    path = np.empty(T, dtype=np.int64)
    j = int(delta.argmax())
    for t in range(T - 1, -1, -1):
        path[t] = j
        if t > 0:
            j = back[t - 1][j]
    return pre.arc[rs[:-1] + path]
