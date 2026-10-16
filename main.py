'''
# ================= main.py =================
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无桌面也能存图；有桌面想看窗口就改成 plt.show()
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 1. 指定默认字体为黑体（Windows常用），也可换成 'Microsoft YaHei'（微软雅黑）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 从我们刚建的 planner.py 里导入两个东西
from gwo_path_planner import plan_path, GWOConfig

# ------------------ 场景参数 ------------------
MAP_SIZE = 100.0
START = (5.0, 5.0)
GOAL  = (95.0, 95.0)
BOUNDS = (0.0, 0.0, MAP_SIZE, MAP_SIZE)   # (xmin, ymin, xmax, ymax)

# 矩形障碍：每条是 (xmin, ymin, xmax, ymax)
RECTANGLES = [
    (30, 20, 45, 55),
    (60, 60, 80, 90),
]

# 圆形障碍：每条是 ((圆心x, 圆心y), 半径)
CIRCLES = [
    ((20, 70), 8.0),
    ((75, 30), 6.0),
]

# ------------------ 算法参数 ------------------
cfg = GWOConfig(
    n_waypoints=18,     # 中间航点数量
    n_wolves=36,        # 狼群数量
    iterations=250,     # 迭代次数
    clearance=2.5,      # 安全距离（无人机半径 + 安全裕度）
    length_weight=1.0,
    turn_weight=0.15,
    collision_weight=1e7,
    seed=7,             # 随机种子，保证结果可复现
)

# ------------------ 调用第二段：规划 ------------------
print("开始规划，请稍候……")
path, score = plan_path(START, GOAL, BOUNDS, RECTANGLES, CIRCLES, cfg)
print("规划完成！")

# ------------------ 统计信息 ------------------
seg = np.diff(path, axis=0)
length = float(np.linalg.norm(seg, axis=1).sum())
print(f"路径总长度: {length:.2f}")
print(f"适应度得分: {score:.2f}")
print(f"航点数量: {len(path)}")

# ------------------ 绘图（支持矩形 + 圆） ------------------
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
ax.set_aspect("equal")

# 画矩形障碍
for (x0, y0, x1, y1) in RECTANGLES:
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                               color="gray", alpha=0.75))
    # 膨胀圈（安全距离外扩）
    e = cfg.clearance
    ax.add_patch(plt.Rectangle((x0-e, y0-e), (x1-x0)+2*e, (y1-y0)+2*e,
                               fill=False, linestyle="--",
                               edgecolor="orange", alpha=0.5))

# 画圆形障碍
for (c, r) in CIRCLES:
    ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.75))
    ax.add_patch(plt.Circle(c, r + cfg.clearance,
                            fill=False, linestyle="--",
                            edgecolor="orange", alpha=0.5))

# 画路径
ax.plot(path[:, 0], path[:, 1], "r.-", linewidth=2.5,
        markersize=5, label="规划路径")
ax.scatter(*START, s=140, c="green", marker="o", zorder=5, label="起点")
ax.scatter(*GOAL,  s=140, c="blue",  marker="s", zorder=5, label="终点")

ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title(f"GWO 无人机路径规划（长度 {length:.1f}）")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("gwo_path_result.png", dpi=160)
print("已保存图片: gwo_path_result.png")
'''
# ================= main.py =================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gwo_path_planner import plan_path, GWOConfig

# ------------------ 场景参数 ------------------
MAP_SIZE = 100.0
START = (5.0, 5.0)
GOAL  = (95.0, 95.0)
BOUNDS = (0.0, 0.0, MAP_SIZE, MAP_SIZE)

# 矩形障碍：(xmin, ymin, xmax, ymax)
RECTANGLES = [
    (30, 20, 45, 55),
    (60, 60, 80, 90),
]

# 圆形障碍：((cx, cy), r)
CIRCLES = [
    ((20, 70), 8.0),
    ((75, 30), 6.0),
]

# 三角形障碍：((x1,y1), (x2,y2), (x3,y3))
TRIANGLES = [
    ((15, 40), (30, 40), (22, 55)),
    ((55, 15), (75, 15), (65, 35)),
    ((45, 75), (60, 85), (40, 90)),
]

# ------------------ 算法参数 ------------------
cfg = GWOConfig(
    n_waypoints=18,
    n_wolves=36,
    iterations=250,
    clearance=2.5,       # 无人机半径 + 安全裕度
    length_weight=1.0,
    turn_weight=0.15,
    collision_weight=1e7,
    seed=7,
)

# ------------------ 规划 ------------------
print("开始规划，请稍候……")
path, score = plan_path(START, GOAL, BOUNDS,
                        rectangles=RECTANGLES,
                        circles=CIRCLES,
                        triangles=TRIANGLES,
                        config=cfg)
print("规划完成！")

seg = np.diff(path, axis=0)
length = float(np.linalg.norm(seg, axis=1).sum())
print(f"路径总长度: {length:.2f}")
print(f"适应度得分: {score:.2f}")
print(f"航点数量: {len(path)}")

# ------------------ 绘图 ------------------
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
ax.set_aspect("equal")

e = cfg.clearance

# 矩形障碍 + 膨胀外框
for (x0, y0, x1, y1) in RECTANGLES:
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                               color="gray", alpha=0.75))
    ax.add_patch(plt.Rectangle((x0-e, y0-e), (x1-x0)+2*e, (y1-y0)+2*e,
                               fill=False, linestyle="--",
                               edgecolor="orange", alpha=0.5))

# 圆形障碍 + 膨胀圈
for (c, r) in CIRCLES:
    ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.75))
    ax.add_patch(plt.Circle(c, r + e, fill=False, linestyle="--",
                            edgecolor="orange", alpha=0.5))

# 三角形障碍 + 膨胀（用外扩多边形近似）
for tri in TRIANGLES:
    tri = np.asarray(tri, float)
    ax.add_patch(plt.Polygon(tri, closed=True, color="gray", alpha=0.75))
    c = tri.mean(axis=0)
    outer = []
    for v in tri:
        d = v - c
        n = np.linalg.norm(d)
        outer.append(v + (d/n)*(e + 0.6) if n > 1e-9 else v.copy())
    ax.add_patch(plt.Polygon(np.asarray(outer), closed=True,
                             fill=False, linestyle="--",
                             edgecolor="orange", alpha=0.5))

# 路径 + 起终点
ax.plot(path[:, 0], path[:, 1], "r.-", linewidth=2.5,
        markersize=5, label="规划路径")
ax.scatter(*START, s=140, c="green", marker="o", zorder=5, label="起点")
ax.scatter(*GOAL,  s=140, c="blue",  marker="s", zorder=5, label="终点")

ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title(f"GWO 无人机路径规划（长度 {length:.1f}）")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("gwo_path_result.png", dpi=160)
print("已保存图片: gwo_path_result.png")
'''
(33.5,60.7,43.6,72.1),
(66.2,18.4,80.9,32.0),

((18.3,47.6),6.1),
((74.5,72.8),5.4),

((44.2,30.5),(52.7,26.1),(48.9,39.8)),
'''