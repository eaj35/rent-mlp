#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月租预测 MLP —— 数据构建 + 训练 + 导出

数据来源（全部为网络公开材料）：
  A. 真实房源：上海链家租房 10,998 条（GitHub 开源，2020-10 抓取）
     仓库 https://github.com/XiyueWang/shanghai_rental_analysis_lianjia
     字段由该仓库 scraper/scraper.py 定义，无表头，共 21 列。
  B. 城市租金水平：35 城住宅租金均价（2025 上半年，元/㎡/月）
     《财经》https://news.caijingmobile.com/article/detail/552055
     中指研究院 50 城报告 https://www.cih-index.com/news/2025-04-23/52652700.html
  C. 影响因素实证系数（用于校验模型学到的方向是否合理）：
     武汉/杭州/长沙/广州 特征价格模型研究，见 HTML 页面「参考资料」。

重要说明：
  单一城市样本无法体现「城市物价水平」。因此以真实上海房源为结构骨架，
  用公开的城市租金指数把样本扩展到多个城市（价格按指数比例缩放）。
  这属于「真实房源结构 + 公开统计标定的扩展样本」，不是可直接用于
  商业定价的真实成交数据集。页面上会明确标注，不冒充真实多城市数据。

输出：
  model.json  —— 网络权重、标准化参数、训练曲线、评估指标
"""
import csv
import json
import math
import os
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "lianjia_shanghai.csv")

# 上海人民广场（作为市中心基准点）
CENTER_LON, CENTER_LAT = 121.4737, 31.2304

# 21 列字段名（依据 scraper.py 的写出顺序）
COLS = [
    "community", "rent_type", "beds", "price", "lon", "lat",
    "tag_balcony", "tag_pay1", "tag_apartment", "tag_monthly", "tag_anytime",
    "tag_subway", "tag_bathroom", "tag_fine_deco", "tag_new",
    "area", "orientation", "level", "elevator", "parking", "amenities",
]

# 城市租金水平（元/㎡/月）。标注 exact 的来自《财经》35 城表；
# 其余按中指研究院「一线 83.34 / 二线 29.89 / 三四线 24.45」分档估计。
CITY_INDEX = {
    "北京": (117.2, "exact"), "上海": (104.4, "exact"), "深圳": (96.8, "exact"),
    "杭州": (62.5, "exact"), "广州": (59.5, "exact"), "南京": (54.2, "exact"),
    "厦门": (45.0, "tier"), "三亚": (48.0, "tier"), "苏州": (34.0, "tier"),
    "宁波": (32.0, "tier"), "福州": (32.0, "tier"), "天津": (32.0, "tier"),
    "武汉": (30.0, "tier"), "青岛": (29.0, "tier"), "大连": (28.0, "tier"),
    "成都": (28.0, "tier"), "西安": (26.0, "tier"), "合肥": (26.0, "tier"),
    "长沙": (25.0, "tier"), "重庆": (25.0, "tier"), "昆明": (24.0, "tier"),
    "南昌": (23.0, "tier"), "南宁": (23.0, "tier"), "兰州": (22.0, "tier"),
    "沈阳": (22.0, "tier"), "石家庄": (22.0, "exact"), "郑州": (21.8, "exact"),
    "贵阳": (21.8, "exact"), "长春": (21.0, "tier"), "哈尔滨": (21.0, "tier"),
    "乌鲁木齐": (21.0, "tier"), "太原": (20.3, "exact"),
    "呼和浩特": (20.1, "exact"), "南通": (20.0, "tier"),
    "惠州": (19.0, "tier"), "西宁": (20.0, "tier"), "银川": (15.5, "exact"),
}
SH_INDEX = CITY_INDEX["上海"][0]


def haversine(lon, lat):
    """经纬度 → 距市中心公里数"""
    r = 6371.0
    p1, p2 = math.radians(lat), math.radians(CENTER_LAT)
    dp = p2 - p1
    dl = math.radians(CENTER_LON - lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_floor(level_text):
    """'中楼层/16层' → (所在楼层估计值, 总楼层)"""
    try:
        head, total = level_text.split("/")
        total = int(total.replace("层", "").strip())
        factor = {"低楼层": 0.2, "中楼层": 0.5, "高楼层": 0.8}.get(head.strip(), 0.5)
        return max(1, round(total * factor)), total
    except Exception:
        return None, None


def load_real_rows():
    rows = []
    with open(RAW, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f, fieldnames=COLS):
            try:
                price = float(r["price"])
                area = float(r["area"])
                lon, lat = float(r["lon"]), float(r["lat"])
                floor, total = parse_floor(r["level"])
                if floor is None:
                    continue
                if not (1000 <= price <= 40000 and 10 <= area <= 250):
                    continue
                if not (120.8 < lon < 122.1 and 30.6 < lat < 31.9):
                    continue
                d = haversine(lon, lat)
                if d > 45:
                    continue
                # 装修程度：精装标签=3；否则按电梯、配套数给 1~2
                fine = int(r["tag_fine_deco"]) == 1
                amen = int(float(r["amenities"])) if r["amenities"] else 0
                if fine:
                    deco = 3.0
                elif r["elevator"] == "有" and amen >= 3:
                    deco = 2.0
                else:
                    deco = 1.0
                rows.append({
                    "dist": d, "deco": deco, "floor": float(floor),
                    "total_floor": float(total), "area": area,
                    "price": price, "subway": int(r["tag_subway"]) == 1,
                })
            except (ValueError, TypeError, KeyError):
                continue
    return rows


def build_dataset(seed=42):
    """真实房源为骨架，按公开城市租金指数扩展到多城市"""
    rng = random.Random(seed)
    real = load_real_rows()
    cities = list(CITY_INDEX)
    # 让上海样本占比高一些，其余城市均匀
    weighted = ["上海"] * 3 + cities
    data = []
    for r in real:
        city = rng.choice(weighted)
        idx = CITY_INDEX[city][0]
        scale = idx / SH_INDEX
        # 注意：真实房源价格里已经包含距市中心的距离效应
        #（实测 单价 与 距离 的相关系数为 -0.71），此处不能再叠加一次，
        # 否则会重复计算。城市维度只做整体价格水平缩放。
        price = r["price"] * scale
        price *= rng.gauss(1.0, 0.05)          # 市场噪声
        price = max(500, price)
        data.append({
            "x": [r["dist"], r["deco"], r["floor"], idx, r["area"]],
            "y": price,
            "city": city,
        })
    return data, len(real)


class MLP:
    """5 → 24 → 16 → 1，ReLU 隐藏层，Adam 优化，MSE 损失（纯 numpy）"""

    def __init__(self, dims, seed=7):
        rs = np.random.RandomState(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            self.W.append(rs.randn(fan_in, fan_out) * math.sqrt(2.0 / fan_in))
            self.b.append(np.zeros((1, fan_out)))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        self.cache = [X]
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(0, z)
            self.cache.append((z, a))
        return a @ self.W[-1] + self.b[-1]

    def backward(self, pred, y):
        n = y.shape[0]
        d = 2.0 * (pred - y) / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        gW[-1] = self.cache[-1][1].T @ d
        gb[-1] = d.sum(axis=0, keepdims=True)
        d = d @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            z, _ = self.cache[i + 1]
            dz = d * (z > 0)
            gW[i] = self.cache[i].T @ dz if i == 0 else self.cache[i][1].T @ dz
            gb[i] = dz.sum(axis=0, keepdims=True)
            d = dz @ self.W[i].T
        return gW, gb

    def step(self, gW, gb, lr, t):
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            for m, v, g, p in (
                (self.mW[i], self.vW[i], gW[i], self.W[i]),
                (self.mb[i], self.vb[i], gb[i], self.b[i]),
            ):
                m *= b1; m += (1 - b1) * g
                v *= b2; v += (1 - b2) * (g * g)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)


def main():
    data, n_real = build_dataset()
    print(f"真实房源清洗后: {n_real} 条 → 扩展样本: {len(data)} 条")

    rs = np.random.RandomState(1)
    rs.shuffle(data)
    X = np.array([d["x"] for d in data], dtype=np.float64)
    y = np.array([d["y"] for d in data], dtype=np.float64).reshape(-1, 1)

    n_train = int(len(X) * 0.8)
    Xtr, Xva = X[:n_train], X[n_train:]
    ytr, yva = y[:n_train], y[n_train:]

    x_mean, x_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = ytr.mean(), ytr.std()
    Xtr_s, Xva_s = (Xtr - x_mean) / x_std, (Xva - x_mean) / x_std
    ytr_s, yva_s = (ytr - y_mean) / y_std, (yva - y_mean) / y_std

    net = MLP([5, 24, 16, 1])
    curve = []
    epochs, batch, lr, t = 900, 256, 0.004, 0
    for ep in range(1, epochs + 1):
        idx = np.random.RandomState(ep).permutation(len(Xtr_s))
        for s in range(0, len(idx), batch):
            k = idx[s:s + batch]
            pred = net.forward(Xtr_s[k])
            gW, gb = net.backward(pred, ytr_s[k])
            t += 1
            net.step(gW, gb, lr, t)
        if ep % 15 == 0 or ep == 1:
            tr = float(np.mean((net.forward(Xtr_s) - ytr_s) ** 2))
            va = float(np.mean((net.forward(Xva_s) - yva_s) ** 2))
            curve.append([ep, round(tr, 5), round(va, 5)])

    def predict(Xs):
        return net.forward(Xs) * y_std + y_mean

    ptr, pva = predict(Xtr_s), predict(Xva_s)
    mae = float(np.mean(np.abs(pva - yva)))
    rmse = float(np.sqrt(np.mean((pva - yva) ** 2)))
    r2 = float(1 - np.sum((yva - pva) ** 2) / np.sum((yva - yva.mean()) ** 2))
    mape = float(np.mean(np.abs((pva - yva) / np.maximum(yva, 1))) * 100)

    print(f"验证集  R²={r2:.4f}  MAE={mae:.0f} 元  RMSE={rmse:.0f} 元  MAPE={mape:.1f}%")
    print(f"月租范围  真实样本 {y.min():.0f} ~ {y.max():.0f} 元")

    # 单因素敏感性（把其余特征固定在中位数，看每个特征的影响方向）
    med = np.median(Xtr, axis=0)
    sens = {}
    for j, name in enumerate(["距市中心(km)", "装修程度", "楼层", "城市物价", "面积"]):
        lo, hi = np.percentile(Xtr[:, j], 10), np.percentile(Xtr[:, j], 90)
        a, b = med.copy(), med.copy()
        a[j], b[j] = lo, hi
        pa = float(predict(((a - x_mean) / x_std).reshape(1, -1))[0, 0])
        pb = float(predict(((b - x_mean) / x_std).reshape(1, -1))[0, 0])
        sens[name] = {"low": round(float(lo), 2), "high": round(float(hi), 2),
                      "rent_low": round(pa), "rent_high": round(pb),
                      "delta": round(pb - pa)}

    model = {
        "meta": {
            "arch": "5-24-16-1", "activation": "ReLU", "optimizer": "Adam",
            "loss": "MSE", "epochs": epochs, "batch": batch, "lr": lr,
            "train_size": n_train, "val_size": len(Xva),
            "real_listings": n_real, "total_samples": len(data),
            "features": ["距市中心(km)", "装修程度(0-3)", "所在楼层", "城市物价水平(元/㎡/月)", "面积(㎡)"],
            "target": "月租(元/月)",
        },
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": float(y_mean), "y_std": float(y_std),
        "W": [w.tolist() for w in net.W], "b": [b.tolist() for b in net.b],
        "metrics": {"r2": round(r2, 4), "mae": round(mae), "rmse": round(rmse),
                    "mape": round(mape, 2)},
        "curve": curve, "sensitivity": sens,
        "city_index": {k: v[0] for k, v in CITY_INDEX.items()},
    }
    out = os.path.join(HERE, "model.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print(f"已导出 {out}")

    print("\n单因素敏感性（其余特征固定在中位数）：")
    for k, v in sens.items():
        print(f"  {k:24s} {v['low']:>8} → {v['high']:<8} 月租 {v['rent_low']:>6} → {v['rent_high']:<6} ({v['delta']:+d})")


if __name__ == "__main__":
    main()
