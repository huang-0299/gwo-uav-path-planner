"""Collision-aware Grey Wolf path planner.

The planner uses continuous segment collision tests (rather than checking only
the sampled waypoints), inflates obstacles by the vehicle clearance, seeds the
population with a visibility-graph path, and repairs infeasible wolves after
each GWO update.  Obstacles are dictionaries/tuples accepted by
``plan_path``: rectangles ``(xmin, ymin, xmax, ymax)`` and circles
``((x, y), radius)``.
"""
'''
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def segment_rect_hit(a, b, rect, eps=1e-10):
    """Liang--Barsky segment/AABB test; catches crossings between waypoints."""
    x0, y0 = map(float, a); x1, y1 = map(float, b)
    xmin, ymin, xmax, ymax = map(float, rect)
    if xmin > xmax: xmin, xmax = xmax, xmin
    if ymin > ymax: ymin, ymax = ymax, ymin
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) <= eps:
            if q < 0: return False
            continue
        t = q / p
        if p < 0: t0 = max(t0, t)
        else: t1 = min(t1, t)
        if t0 - t1 > eps: return False
    return True


def segment_circle_hit(a, b, center, radius, margin=0.0):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    c = np.asarray(center, dtype=float); v = b - a
    den = float(v @ v)
    t = 0.0 if den <= 1e-15 else float(np.clip((c - a) @ v / den, 0, 1))
    return float(np.linalg.norm(a + t * v - c)) <= float(radius) + margin


def path_hits(path, rectangles=(), circles=(), clearance=0.0):
    for a, b in zip(path[:-1], path[1:]):
        for r in rectangles:
            x0, y0, x1, y1 = map(float, r)
            if segment_rect_hit(a, b, (x0-clearance, y0-clearance,
                                       x1+clearance, y1+clearance)):
                return True
        for c, r in circles:
            if segment_circle_hit(a, b, c, r, clearance): return True
    return False


def _length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _resample(polyline, n_points):
    """Resample a polyline to exactly n_points by arc length."""
    p = np.asarray(polyline, dtype=float)
    if len(p) == n_points: return p.copy()
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    if s[-1] <= 1e-12: return np.repeat(p[:1], n_points, axis=0)
    q = np.linspace(0, s[-1], n_points)
    out = np.empty((n_points, 2), dtype=float)
    for j in range(2): out[:, j] = np.interp(q, s, p[:, j])
    return out


def visibility_seed(start, goal, bounds, rectangles, circles, clearance):
    """Shortest collision-free polyline on obstacle-corner visibility graph."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    nodes = [np.asarray(start, float), np.asarray(goal, float)]
    e = float(clearance) + 1e-4
    for x0, y0, x1, y1 in rectangles:
        nodes += [np.array([x0-e, y0-e]), np.array([x0-e, y1+e]),
                  np.array([x1+e, y0-e]), np.array([x1+e, y1+e])]
    # Four tangent-ish points provide useful detours around circular obstacles.
    for c, r in circles:
        c = np.asarray(c, float); rr = float(r) + e
        nodes += [c + rr*np.array([1., 0.]), c + rr*np.array([-1., 0.]),
                  c + rr*np.array([0., 1.]), c + rr*np.array([0., -1.])]
    nodes = [np.clip(p, [xmin, ymin], [xmax, ymax]) for p in nodes]
    n = len(nodes); g = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not path_hits([nodes[i], nodes[j]], rectangles, circles, clearance):
                w = float(np.linalg.norm(nodes[i]-nodes[j]))
                g[i].append((j, w)); g[j].append((i, w))
    dist = [float('inf')]*n; prev = [-1]*n; dist[0] = 0.; h = [(0.0, 0)]
    while h:
        d, i = heapq.heappop(h)
        if d != dist[i]: continue
        for j, w in g[i]:
            nd = d + w
            if nd < dist[j]: dist[j] = nd; prev[j] = i; heapq.heappush(h, (nd, j))
    if not np.isfinite(dist[1]): return np.vstack([start, goal])
    ids = []; i = 1
    while i >= 0: ids.append(i); i = prev[i]
    return np.vstack([nodes[i] for i in ids[::-1]])


@dataclass
class GWOConfig:
    n_waypoints: int = 18
    n_wolves: int = 36
    iterations: int = 250
    clearance: float = 1.0
    length_weight: float = 1.0
    turn_weight: float = 0.15
    collision_weight: float = 1e7
    seed: int | None = 7


class GreyWolfPlanner:
    def __init__(self, start, goal, bounds, rectangles=(), circles=(), config=None):
        self.start = np.asarray(start, float); self.goal = np.asarray(goal, float)
        self.bounds = np.asarray(bounds, float); self.rectangles = tuple(rectangles)
        self.circles = tuple(circles); self.cfg = config or GWOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dim = 2 * self.cfg.n_waypoints

    def _path(self, x):
        return np.vstack([self.start, np.asarray(x).reshape(-1, 2), self.goal])

    def fitness(self, x):
        p = self._path(x)
        hits = 0; penetration = 0.
        # Count all intersecting segments so a collision-free alpha always wins.
        for a, b in zip(p[:-1], p[1:]):
            for r in self.rectangles:
                rr = (r[0]-self.cfg.clearance, r[1]-self.cfg.clearance,
                      r[2]+self.cfg.clearance, r[3]+self.cfg.clearance)
                if segment_rect_hit(a, b, rr): hits += 1
            for c, r in self.circles:
                if segment_circle_hit(a, b, c, r, self.cfg.clearance): hits += 1
        length = _length(p)
        v = np.diff(p, axis=0); turn = 0.
        for u, w in zip(v[:-1], v[1:]):
            nu = np.linalg.norm(u); nw = np.linalg.norm(w)
            if nu > 1e-12 and nw > 1e-12:
                turn += float(np.arccos(np.clip((u@w)/(nu*nw), -1, 1))**2)
        return self.cfg.length_weight*length + self.cfg.turn_weight*turn + self.cfg.collision_weight*hits

    def repair(self, x):
        p = self._path(x); e = self.cfg.clearance + 1e-3
        # Add obstacle-corner detours for intersecting segments, then resample.
        for _ in range(3):
            changed = False; out = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                candidates = []
                for r in self.rectangles:
                    rr = (r[0]-e, r[1]-e, r[2]+e, r[3]+e)
                    if segment_rect_hit(a, b, rr):
                        x0,y0,x1,y1 = rr
                        candidates += [np.array([x0,y0]),np.array([x0,y1]),
                                       np.array([x1,y0]),np.array([x1,y1])]
                for c, r in self.circles:
                    if segment_circle_hit(a, b, c, r, e):
                        c=np.asarray(c,float); q=r+e
                        candidates += [c+q*np.array([1.,0.]),c+q*np.array([-1.,0.]),
                                       c+q*np.array([0.,1.]),c+q*np.array([0.,-1.])]
                if candidates:
                    cand = min(candidates, key=lambda q: _length(np.vstack([a,q,b]))
                               if not path_hits([a,q,b], self.rectangles,self.circles,self.cfg.clearance) else 1e30)
                    if not path_hits([a, cand, b], self.rectangles,self.circles,self.cfg.clearance):
                        out.append(np.clip(cand, self.bounds[:2], self.bounds[2:])); changed = True
                out.append(b)
            p = np.asarray(out)
            if not changed or not path_hits(p, self.rectangles, self.circles, self.cfg.clearance): break
        p = _resample(p, self.cfg.n_waypoints+2); p[0] = self.start; p[-1] = self.goal
        return p[1:-1].reshape(-1)

    def plan(self):
        lo, hi = self.bounds[:2], self.bounds[2:]
        seed = _resample(visibility_seed(self.start,self.goal,self.bounds,
                                          self.rectangles,self.circles,self.cfg.clearance),
                          self.cfg.n_waypoints+2)[1:-1].reshape(-1)
        low = np.tile(lo, self.cfg.n_waypoints)
        high = np.tile(hi, self.cfg.n_waypoints)
        wolves = self.rng.uniform(low, high, (self.cfg.n_wolves, self.dim))
        wolves[0] = seed
        wolves = np.asarray([self.repair(x) for x in wolves])
        best_x = wolves[0].copy(); best_f = self.fitness(best_x)
        for it in range(self.cfg.iterations):
            fs = np.asarray([self.fitness(x) for x in wolves]); order = np.argsort(fs)
            if fs[order[0]] < best_f: best_f, best_x = float(fs[order[0]]), wolves[order[0]].copy()
            alpha, beta, delta = wolves[order[0]], wolves[order[1]], wolves[order[2]]
            a = 2.0 * (1.0 - it / max(1, self.cfg.iterations-1))
            nxt = np.empty_like(wolves)
            for k, x in enumerate(wolves):
                acc = 0.
                for leader in (alpha, beta, delta):
                    A = 2*a*self.rng.random(self.dim)-a; C = 2*self.rng.random(self.dim)
                    acc += leader - A*np.abs(C*leader-x)
                # Waypoint variables are flattened as [x0,y0,x1,y1,...].
                nxt[k] = np.clip(acc/3, np.tile(lo, self.cfg.n_waypoints),
                                 np.tile(hi, self.cfg.n_waypoints))
            wolves = np.asarray([self.repair(x) for x in nxt])
        # The last GWO move may contain a new optimum; include it before return.
        fs = np.asarray([self.fitness(x) for x in wolves])
        j = int(np.argmin(fs))
        if fs[j] < best_f:
            best_f, best_x = float(fs[j]), wolves[j].copy()
        return self._path(best_x), best_f


def plan_path(start, goal, bounds, rectangles=(), circles=(), config=None):
    return GreyWolfPlanner(start, goal, bounds, rectangles, circles, config).plan()
'''
# ================= planner.py =================
from dataclasses import dataclass
import heapq
import numpy as np


# ---------- 基本几何工具 ----------
def _point_in_triangle(p, tri, eps=1e-9):
    """判断点 p 是否在三角形 tri 内部（含边界）。"""
    p = np.asarray(p, float)
    a, b, c = np.asarray(tri[0], float), np.asarray(tri[1], float), np.asarray(tri[2], float)

    def cross(o, u, v):
        return (u[0]-o[0])*(v[1]-o[1]) - (u[1]-o[1])*(v[0]-o[0])

    d1 = cross(a, b, p)
    d2 = cross(b, c, p)
    d3 = cross(c, a, p)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def _seg_seg_dist(p1, p2, q1, q2):
    """两条线段 (p1,p2) 和 (q1,q2) 的最短距离。"""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - q1))
    if a <= EPS:
        s = 0.0
        t = float(np.clip(f / e, 0, 1))
    else:
        c = float(d1 @ r)
        if e <= EPS:
            t = 0.0
            s = float(np.clip(-c / a, 0, 1))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom > EPS:
                s = float(np.clip((b * f - c * e) / denom, 0, 1))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0:
                t = 0.0
                s = float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t = 1.0
                s = float(np.clip((b - c) / a, 0, 1))
    c1 = p1 + s * d1
    c2 = q1 + t * d2
    return float(np.linalg.norm(c1 - c2))


def segment_rect_hit(a, b, rect, eps=1e-10):
    """Liang-Barsky：线段与 AABB 矩形是否相交。"""
    x0, y0 = map(float, a); x1, y1 = map(float, b)
    xmin, ymin, xmax, ymax = map(float, rect)
    if xmin > xmax: xmin, xmax = xmax, xmin
    if ymin > ymax: ymin, ymax = ymax, ymin
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) <= eps:
            if q < 0: return False
            continue
        t = q / p
        if p < 0: t0 = max(t0, t)
        else: t1 = min(t1, t)
        if t0 - t1 > eps: return False
    return True


def segment_circle_hit(a, b, center, radius, margin=0.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(center, float); v = b - a
    den = float(v @ v)
    t = 0.0 if den <= 1e-15 else float(np.clip((c - a) @ v / den, 0, 1))
    return float(np.linalg.norm(a + t * v - c)) <= float(radius) + margin


def segment_triangle_hit(a, b, tri, margin=0.0):
    """线段到（带 margin 的）三角形的最短距离是否 <= margin。
    等价于：三角形按 margin 膨胀后与线段相交。
    - 端点落在三角形内  -> 命中
    - 线段到任意一条边的最短距离 <= margin -> 命中
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    tri = np.asarray(tri, float)
    # 端点在三角形内
    if _point_in_triangle(a, tri) or _point_in_triangle(b, tri):
        return True
    # 线段到三角形三条边的距离
    for i in range(3):
        p = tri[i]
        q = tri[(i + 1) % 3]
        if _seg_seg_dist(a, b, p, q) <= margin:
            return True
    return False


def path_hits(path, rectangles=(), circles=(), triangles=(), clearance=0.0):
    for a, b in zip(path[:-1], path[1:]):
        for r in rectangles:
            x0, y0, x1, y1 = map(float, r)
            if segment_rect_hit(a, b, (x0-clearance, y0-clearance,
                                       x1+clearance, y1+clearance)):
                return True
        for c, r in circles:
            if segment_circle_hit(a, b, c, r, clearance):
                return True
        for tri in triangles:
            if segment_triangle_hit(a, b, tri, clearance):
                return True
    return False


def _length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _resample(polyline, n_points):
    p = np.asarray(polyline, dtype=float)
    if len(p) == n_points: return p.copy()
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    if s[-1] <= 1e-12: return np.repeat(p[:1], n_points, axis=0)
    q = np.linspace(0, s[-1], n_points)
    out = np.empty((n_points, 2), dtype=float)
    for j in range(2): out[:, j] = np.interp(q, s, p[:, j])
    return out


def visibility_seed(start, goal, bounds, rectangles, circles, triangles, clearance):
    """把障碍物角点当作图节点，做最短路，得到一条较短的初始路径。"""
    xmin, ymin, xmax, ymax = map(float, bounds)
    nodes = [np.asarray(start, float), np.asarray(goal, float)]
    e = float(clearance) + 1e-4

    # 矩形：四角外推
    for x0, y0, x1, y1 in rectangles:
        nodes += [np.array([x0-e, y0-e]), np.array([x0-e, y1+e]),
                  np.array([x1+e, y0-e]), np.array([x1+e, y1+e])]

    # 圆：四个方向外推
    for c, r in circles:
        c = np.asarray(c, float); rr = float(r) + e
        nodes += [c + rr*np.array([1., 0.]), c + rr*np.array([-1., 0.]),
                  c + rr*np.array([0., 1.]), c + rr*np.array([0., -1.])]

    # 三角形：三个顶点沿“质心 -> 顶点”方向外推
    for tri in triangles:
        tri = np.asarray(tri, float)
        c = tri.mean(axis=0)
        for v in tri:
            d = v - c
            n = np.linalg.norm(d)
            if n > 1e-9:
                nodes.append(v + (d / n) * e)
            else:
                nodes.append(v.copy())

    nodes = [np.clip(p, [xmin, ymin], [xmax, ymax]) for p in nodes]
    n = len(nodes); g = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not path_hits([nodes[i], nodes[j]],
                             rectangles, circles, triangles, clearance):
                w = float(np.linalg.norm(nodes[i] - nodes[j]))
                g[i].append((j, w)); g[j].append((i, w))

    dist = [float('inf')] * n; prev = [-1] * n
    dist[0] = 0.0; h = [(0.0, 0)]
    while h:
        d, i = heapq.heappop(h)
        if d != dist[i]: continue
        for j, w in g[i]:
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd; prev[j] = i
                heapq.heappush(h, (nd, j))
    if not np.isfinite(dist[1]):
        return np.vstack([start, goal])
    ids = []; i = 1
    while i >= 0: ids.append(i); i = prev[i]
    return np.vstack([nodes[i] for i in ids[::-1]])


@dataclass
class GWOConfig:
    n_waypoints: int = 18
    n_wolves: int = 36
    iterations: int = 250
    clearance: float = 1.0
    length_weight: float = 1.0
    turn_weight: float = 0.15
    collision_weight: float = 1e7
    seed: int | None = 7


class GreyWolfPlanner:
    def __init__(self, start, goal, bounds, rectangles=(), circles=(),
                 triangles=(), config=None):
        self.start = np.asarray(start, float);
        self.goal = np.asarray(goal, float)
        self.bounds = np.asarray(bounds, float)
        self.rectangles = tuple(rectangles)
        self.circles = tuple(circles)
        self.triangles = tuple(tuple(map(tuple, tri)) for tri in triangles)  # 3x2
        self.cfg = config or GWOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dim = 2 * self.cfg.n_waypoints

    def _path(self, x):
        return np.vstack([self.start, np.asarray(x).reshape(-1, 2), self.goal])

    def fitness(self, x):
        p = self._path(x)
        hits = 0
        for a, b in zip(p[:-1], p[1:]):
            for r in self.rectangles:
                rr = (r[0] - self.cfg.clearance, r[1] - self.cfg.clearance,
                      r[2] + self.cfg.clearance, r[3] + self.cfg.clearance)
                if segment_rect_hit(a, b, rr): hits += 1
            for c, r in self.circles:
                if segment_circle_hit(a, b, c, r, self.cfg.clearance): hits += 1
            for tri in self.triangles:
                if segment_triangle_hit(a, b, np.asarray(tri, float),
                                        self.cfg.clearance):
                    hits += 1

        length = _length(p)
        v = np.diff(p, axis=0);
        turn = 0.0
        for u, w in zip(v[:-1], v[1:]):
            nu = np.linalg.norm(u);
            nw = np.linalg.norm(w)
            if nu > 1e-12 and nw > 1e-12:
                turn += float(np.arccos(np.clip((u @ w) / (nu * nw), -1, 1)) ** 2)
        return self.cfg.length_weight * length \
            + self.cfg.turn_weight * turn \
            + self.cfg.collision_weight * hits

    def repair(self, x):
        """把有碰撞的路径局部推开：在障碍物角点外推取候选绕行点，再重采样。"""
        p = self._path(x);
        e = self.cfg.clearance + 1e-3
        for _ in range(3):
            changed = False;
            out = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                candidates = []
                # 矩形
                for r in self.rectangles:
                    rr = (r[0] - e, r[1] - e, r[2] + e, r[3] + e)
                    if segment_rect_hit(a, b, rr):
                        x0, y0, x1, y1 = rr
                        candidates += [np.array([x0, y0]), np.array([x0, y1]),
                                       np.array([x1, y0]), np.array([x1, y1])]
                # 圆
                for c, r in self.circles:
                    if segment_circle_hit(a, b, c, r, e):
                        c = np.asarray(c, float);
                        q = r + e
                        candidates += [c + q * np.array([1., 0.]),
                                       c + q * np.array([-1., 0.]),
                                       c + q * np.array([0., 1.]),
                                       c + q * np.array([0., -1.])]
                # 三角形：从质心向三个顶点方向外推
                for tri in self.triangles:
                    tri = np.asarray(tri, float)
                    if segment_triangle_hit(a, b, tri, e):
                        c = tri.mean(axis=0)
                        for v in tri:
                            d = v - c
                            n = np.linalg.norm(d)
                            if n > 1e-9:
                                candidates.append(v + (d / n) * (e + 0.5))
                            else:
                                candidates.append(v.copy())

                if candidates:
                    def safe_len(q):
                        sub = [a, q, b]
                        if path_hits(sub, self.rectangles, self.circles,
                                     self.triangles, self.cfg.clearance):
                            return 1e30
                        return _length(np.vstack(sub))

                    cand = min(candidates, key=safe_len)
                    if safe_len(cand) < 1e29:
                        out.append(np.clip(cand, self.bounds[:2], self.bounds[2:]))
                        changed = True
                out.append(b)
            p = np.asarray(out)
            if not changed or not path_hits(p, self.rectangles, self.circles,
                                            self.triangles, self.cfg.clearance):
                break
        p = _resample(p, self.cfg.n_waypoints + 2)
        p[0] = self.start;p[-1] = self.goal
        return p[1:-1].reshape(-1)


    def plan(self):
        lo, hi = self.bounds[:2], self.bounds[2:]
        seed = _resample(
            visibility_seed(self.start, self.goal, self.bounds,
                                        self.rectangles, self.circles, self.triangles,
                                        self.cfg.clearance),
            self.cfg.n_waypoints + 2)[1:-1].reshape(-1)

        low = np.tile(lo, self.cfg.n_waypoints)
        high = np.tile(hi, self.cfg.n_waypoints)
        wolves = self.rng.uniform(low, high, (self.cfg.n_wolves, self.dim))
        wolves[0] = seed
        wolves = np.asarray([self.repair(x) for x in wolves])

        best_x = wolves[0].copy()
        best_f = self.fitness(best_x)
        for it in range(self.cfg.iterations):
            fs = np.asarray([self.fitness(x) for x in wolves])
            order = np.argsort(fs)
            if fs[order[0]] < best_f:
                best_f, best_x = float(fs[order[0]]), wolves[order[0]].copy()
            alpha, beta, delta = wolves[order[0]], wolves[order[1]], wolves[order[2]]
            a = 2.0 * (1.0 - it / max(1, self.cfg.iterations - 1))
            nxt = np.empty_like(wolves)
            for k, x in enumerate(wolves):
                acc = 0.0
                for leader in (alpha, beta, delta):
                    A = 2 * a * self.rng.random(self.dim) - a
                    C = 2 * self.rng.random(self.dim)
                    acc += leader - A * np.abs(C * leader - x)
                nxt[k] = np.clip(acc / 3, low, high)
            wolves = np.asarray([self.repair(x) for x in nxt])

        fs = np.asarray([self.fitness(x) for x in wolves])
        j = int(np.argmin(fs))
        if fs[j] < best_f:
            best_f, best_x = float(fs[j]), wolves[j].copy()
        return self._path(best_x), best_f
def plan_path(start, goal, bounds, rectangles=(), circles=(), triangles=(),
              config=None):
    return GreyWolfPlanner(start, goal, bounds, rectangles, circles, triangles,
                           config).plan()


