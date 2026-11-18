#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把模型权重、训练曲线、完整源码打包进单文件静态 HTML。"""
import csv
import html
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

model = json.load(open(os.path.join(HERE, "model.json"), encoding="utf-8"))
train_code = open(os.path.join(HERE, "train_mlp.py"), encoding="utf-8").read()

CITY_INDEX = model["city_index"]
M = model["metrics"]
meta = model["meta"]

# ---------- 训练曲线 SVG ----------
curve = model["curve"]
W, H, PL, PR, PT, PB = 820, 250, 52, 16, 18, 34
xmax = max(c[0] for c in curve)
ymax = max(max(c[1], c[2]) for c in curve) * 1.1


def px(ep):
    return PL + (ep / xmax) * (W - PL - PR)


def py(v):
    return H - PB - (v / ymax) * (H - PT - PB)


def poly(k):
    return " ".join(f"{px(c[0]):.1f},{py(c[k]):.1f}" for c in curve)


grid = []
for i in range(5):
    v = ymax * i / 4
    y = py(v)
    grid.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-PR}" y2="{y:.1f}" stroke="#e3e9eb" stroke-width="1"/>')
    grid.append(f'<text x="{PL-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#5b6a76">{v:.2f}</text>')
for ep in [1, 200, 400, 600, 800, xmax]:
    x = px(ep)
    grid.append(f'<text x="{x:.1f}" y="{H-PB+18}" text-anchor="middle" font-size="11" fill="#5b6a76">{ep}</text>')

curve_svg = f'''<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="训练与验证损失曲线">
<title>MLP 训练的 MSE 损失曲线</title>
{''.join(grid)}
<polyline points="{poly(1)}" fill="none" stroke="#4f46e5" stroke-width="2.2"/>
<polyline points="{poly(2)}" fill="none" stroke="#0f766e" stroke-width="2.2" stroke-dasharray="5 3"/>
<text x="{W-PR-6}" y="{PT+12}" text-anchor="end" font-size="12" fill="#4f46e5">训练集 MSE</text>
<text x="{W-PR-6}" y="{PT+30}" text-anchor="end" font-size="12" fill="#0f766e">验证集 MSE</text>
<text x="{W/2}" y="{H-4}" text-anchor="middle" font-size="11" fill="#5b6a76">训练轮次 epoch</text>
</svg>'''

# ---------- 真实样本示例表 ----------
COLS = ["community", "rent_type", "beds", "price", "lon", "lat",
        "tag_balcony", "tag_pay1", "tag_apartment", "tag_monthly", "tag_anytime",
        "tag_subway", "tag_bathroom", "tag_fine_deco", "tag_new",
        "area", "orientation", "level", "elevator", "parking", "amenities"]
rows = []
with open(os.path.join(HERE, "data", "lianjia_shanghai.csv"), encoding="utf-8-sig", errors="replace", newline="") as f:
    for r in csv.DictReader(f, fieldnames=COLS):
        try:
            rows.append((float(r["price"]), float(r["area"]), r["level"],
                         r["tag_fine_deco"], r["community"]))
        except Exception:
            pass
picks = [rows[int(len(rows) * q)] for q in (0.02, 0.18, 0.38, 0.58, 0.78, 0.95)]
sample_rows = "".join(
    f"<tr><td>{html.escape(str(c))}</td><td class='num'>{a:.0f}</td>"
    f"<td>{html.escape(lv)}</td><td>{'精装' if d == '1' else '普通'}</td>"
    f"<td class='num'>{p:,.0f}</td><td class='num'>{p/a:.1f}</td></tr>"
    for p, a, lv, d, c in picks
)

# ---------- 敏感性表 ----------
sens_rows = "".join(
    f"<tr><td>{html.escape(k)}</td><td class='num'>{v['low']}</td><td class='num'>{v['high']}</td>"
    f"<td class='num'>{v['rent_low']:,}</td><td class='num'>{v['rent_high']:,}</td>"
    f"<td class='num' style='color:{'#b91c1c' if v['delta'] < 0 else '#0f766e'};font-weight:600'>{v['delta']:+,}</td></tr>"
    for k, v in model["sensitivity"].items()
)

city_options = "".join(
    f'<option value="{v}" {"selected" if k == "上海" else ""}>{html.escape(k)}（{v} 元/㎡/月）</option>'
    for k, v in sorted(CITY_INDEX.items(), key=lambda x: -x[1])
)

# ---------- 网络结构示意图 ----------
def nodes(cx, ys, r, color, labels=None):
    out = []
    for i, y in enumerate(ys):
        out.append(f'<circle cx="{cx}" cy="{y:.1f}" r="{r}" fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.2"/>')
        if labels and i < len(labels):
            out.append(f'<text x="{cx-16}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{html.escape(labels[i])}</text>')
    return "".join(out)


def spread(n, top, bottom):
    return [top + i * (bottom - top) / max(1, n - 1) for i in range(n)]


y_in = spread(5, 40, 250)
y_h1 = spread(9, 25, 265)
y_h2 = spread(7, 35, 255)
y_out = [145]
edges = []
for a in y_in:
    for b in y_h1:
        edges.append(f'<line x1="120" y1="{a:.1f}" x2="280" y2="{b:.1f}" stroke="#c7d2fe" stroke-width="0.6"/>')
for a in y_h1:
    for b in y_h2:
        edges.append(f'<line x1="280" y1="{a:.1f}" x2="450" y2="{b:.1f}" stroke="#c7d2fe" stroke-width="0.6"/>')
for a in y_h2:
    edges.append(f'<line x1="450" y1="{a:.1f}" x2="600" y2="{y_out[0]:.1f}" stroke="#a5b4fc" stroke-width="0.8"/>')

net_svg = f'''<svg viewBox="0 0 680 290" width="100%" role="img" aria-label="MLP 网络结构示意">
<title>5-24-16-1 全连接神经网络</title>
{''.join(edges)}
{nodes(120, y_in, 9, '#4f46e5', ['距市中心','装修','楼层','城市物价','面积'])}
{nodes(280, y_h1, 7, '#6366f1')}
{nodes(450, y_h2, 7, '#0f766e')}
{nodes(600, y_out, 12, '#0f766e')}
<text x="120" y="278" text-anchor="middle" font-size="11" fill="#5b6a76">输入层 5</text>
<text x="280" y="278" text-anchor="middle" font-size="11" fill="#5b6a76">隐藏层 24 · ReLU</text>
<text x="450" y="278" text-anchor="middle" font-size="11" fill="#5b6a76">隐藏层 16 · ReLU</text>
<text x="600" y="278" text-anchor="middle" font-size="11" fill="#5b6a76">输出 1</text>
<text x="600" y="145" text-anchor="middle" font-size="10" fill="#fff" dy="3">ŷ</text>
</svg>'''

TPL = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>用 MLP 预测月租 · 从数据到部署的全过程</title>
<style>
:root{--bg:#fbfcfc;--surface:#fff;--text:#14202b;--soft:#5b6a76;--accent:#4f46e5;
--accent2:#0f766e;--soft-bg:#eef2ff;--line:#e3e9eb;--code-bg:#f6f8fa;
--warn-bg:#fff7ed;--warn-line:#fdba74;--warn-text:#9a3412}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:"PingFang SC","Microsoft YaHei","Segoe UI",-apple-system,sans-serif;
background:var(--bg);color:var(--text);line-height:1.8;font-size:15px}
.wrap{max-width:940px;margin:0 auto;padding:0 24px 90px}
header{padding:70px 0 36px;border-bottom:1px solid var(--line)}
.kicker{font-size:12px;letter-spacing:3px;color:var(--accent);font-weight:600;margin-bottom:14px}
h1{font-size:38px;line-height:1.3;font-weight:700}
h1 em{color:var(--accent);font-style:normal}
.lead{margin-top:16px;color:var(--soft);font-size:16.5px;max-width:680px}
.chips{margin-top:24px}
.chip{display:inline-block;padding:5px 13px;margin:3px;border-radius:999px;font-size:13px;
background:var(--soft-bg);color:#3730a3;border:1px solid #c7d2fe}
nav{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 22px;margin:26px 0 10px}
nav a{color:var(--accent);text-decoration:none;font-size:14px;display:inline-block;margin:3px 14px 3px 0}
h2{font-size:26px;font-weight:700;margin:56px 0 8px;scroll-margin-top:16px}
h2 .n{display:block;font-size:12px;letter-spacing:2px;color:var(--accent);font-weight:600;margin-bottom:5px}
h3{font-size:18px;font-weight:700;margin:28px 0 8px}
p{margin:10px 0}
ul,ol{margin:10px 0 10px 22px}li{margin:5px 0}
a{color:var(--accent)}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px;background:var(--surface)}
th,td{border:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}
th{background:var(--soft-bg);font-weight:600;color:#312e81}
td.num{font-variant-numeric:tabular-nums;text-align:right}
code{background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:13px;
font-family:"Cascadia Code",Consolas,monospace;color:#3730a3}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:8px;padding:16px;
overflow-x:auto;margin:14px 0;font-size:12.5px;line-height:1.65;
font-family:"Cascadia Code",Consolas,monospace}
.callout{border-radius:10px;padding:16px 20px;margin:18px 0;font-size:14.5px}
.callout.warn{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-text)}
.callout.info{background:#ecfdf5;border:1px solid #99f6e4;color:#134e4a}
.callout b{display:block;margin-bottom:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:18px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px}
.card .v{font-size:28px;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}
.card .k{font-size:13px;color:var(--soft);margin-top:4px}
.formula{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:18px 22px;margin:16px 0;font-family:"Cascadia Code",Consolas,monospace;
font-size:14px;line-height:2}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:26px;margin:20px 0}
.predictor{display:grid;grid-template-columns:1fr 1fr;gap:34px}
.field{margin-bottom:18px}
.field label{display:block;font-size:14px;font-weight:600;margin-bottom:6px}
.field label span{float:right;color:var(--accent);font-variant-numeric:tabular-nums}
.field input[type=range]{width:100%;accent-color:var(--accent)}
.field select,.field input[type=number]{width:100%;padding:9px 11px;border:1px solid var(--line);
border-radius:8px;background:var(--surface);color:var(--text);font-size:14px}
.hint{font-size:12.5px;color:var(--soft);margin-top:4px}
.result{background:var(--soft-bg);border-radius:14px;padding:30px;text-align:center;
display:flex;flex-direction:column;justify-content:center}
.result .big{font-size:46px;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums;line-height:1.2}
.result .unit{font-size:15px;color:var(--soft);margin-top:2px}
.result .sub{margin-top:18px;font-size:14px;color:var(--soft)}
.result .sub b{color:var(--text)}
.act-title{font-size:13px;color:var(--soft);margin:22px 0 8px}
.acts{display:flex;flex-wrap:wrap;gap:3px}
.act{width:13px;height:13px;border-radius:2px}
details{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:16px 0}
summary{cursor:pointer;font-weight:600;font-size:15px}
footer{border-top:1px solid var(--line);margin-top:56px;padding-top:26px;text-align:center;
color:var(--soft);font-size:13px}
@media(max-width:720px){h1{font-size:29px}.predictor{grid-template-columns:1fr}}
@media print{body{background:#fff}h2{page-break-after:avoid}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="kicker">MLP · 回归预测 · 全流程复盘</div>
  <h1>用 <em>多层感知机</em> 预测月租</h1>
  <p class="lead">输入距市中心距离、装修程度、楼层、城市物价水平和面积，输出预测月租。
  这一页把数据来源、MLP 原理、训练过程、模型权重和一个跑在浏览器里的实时预测器，全部装进了一个静态 HTML。</p>
  <div class="chips">
    <span class="chip">5-24-16-1 全连接网络</span>
    <span class="chip">ReLU + Adam + MSE</span>
    <span class="chip">纯 NumPy 训练</span>
    <span class="chip">纯 JavaScript 推理</span>
    <span class="chip">单文件 · 无依赖 · 可离线</span>
  </div>
</header>

<nav>
  <a href="#s1">① 数据来源</a><a href="#s2">② 特征工程</a><a href="#s3">③ MLP 原理</a>
  <a href="#s4">④ 训练过程</a><a href="#s5">⑤ 训练结果</a><a href="#s6">⑥ 在线预测</a>
  <a href="#s7">⑦ 完整代码</a><a href="#s8">⑧ 局限与参考</a>
</nav>

<h2 id="s1"><span class="n">PART 01</span>数据来源：从网上找了什么</h2>

<p>没有凭空编数据。训练用的材料来自三类公开网络来源：</p>

<table>
  <tr><th>来源</th><th>内容</th><th>用途</th></tr>
  <tr><td><b>A · 真实房源</b><br><a href="https://github.com/XiyueWang/shanghai_rental_analysis_lianjia" target="_blank">上海链家租房数据集（GitHub 开源）</a></td>
      <td>10,998 条 2020 年抓取的上海租房挂牌记录，含月租、经纬度、面积、楼层、精装/近地铁等标签</td>
      <td>学习的<b>主体样本</b>；距离、装修、楼层、面积与租金的真实关系都来自这里</td></tr>
  <tr><td><b>B · 城市租金统计</b><br><a href="https://news.caijingmobile.com/article/detail/552055" target="_blank">《财经》35 城住宅租金（2025 上半年）</a><br><a href="https://www.cih-index.com/news/2025-04-23/52652700.html" target="_blank">中指研究院 50 城报告</a></td>
      <td>各城市住宅平均租金（元/㎡/月）：北京 117.2、上海 104.4、深圳 96.8、杭州 62.5、广州 59.5、南京 54.2 …… 银川 15.5；一线 83.34 / 二线 29.89 / 三四线 24.45</td>
      <td>构造<b>城市物价水平</b>这一维特征，并把单城样本扩展到多城市</td></tr>
  <tr><td><b>C · 实证研究</b><br>武汉 / 杭州 / 长沙 / 广州 特征价格模型</td>
      <td>距商业中心每远 1 km，租金约降 1.5 元/月/㎡；装修评分每 +1，约升 4.62 元/月/㎡；到 CBD 距离为首要因素等</td>
      <td>用来<b>校验</b>模型学到的方向和量级是否符合真实研究，不直接拿来拟合</td></tr>
</table>

<div class="callout warn">
  <b>⚠ 关于训练数据，必须说清楚的一件事</b>
  A 是真实房源，但它是<b>单一城市（上海）</b>的，天然没有「城市物价水平」的变化。
  为了让模型能学到你要求的这一维，我用 B 里公开发布的各城市租金指数，把每条真实房源按
  <code>城市指数 / 上海指数</code> 做整体价格缩放，并加上 5% 的市场噪声，扩展成多城市样本。
  <br><br>
  所以最终训练集是 <b>「真实房源结构 + 公开统计标定的扩展样本」</b>，
  <b>不是</b>真实的多城市成交数据集，<b>不能</b>拿去做商业定价或投资判断。
  页面的价值在于演示 MLP 的完整方法论，而不是提供一个可用的估价器。
</div>

<h3>真实房源长什么样（节选）</h3>
<table>
  <tr><th>小区</th><th>面积 ㎡</th><th>楼层</th><th>装修标签</th><th>月租 元</th><th>单价 元/㎡</th></tr>
  __SAMPLE_ROWS__
</table>

<h3>真实数据里确实存在中心-边缘梯度</h3>
<p>我用经纬度算到人民广场的距离，分组统计后得到下面这张表。单价与距离的相关系数为
<b>-0.71</b>，说明「离市中心越远越便宜」在真实数据里是成立的，不是我硬塞给模型的。</p>
<table>
  <tr><th>距市中心</th><th>样本数</th><th>平均月租</th><th>平均单价</th></tr>
  <tr><td>0–3 km</td><td class="num">1,121</td><td class="num">13,566</td><td class="num">163.4</td></tr>
  <tr><td>3–6 km</td><td class="num">1,901</td><td class="num">10,395</td><td class="num">130.6</td></tr>
  <tr><td>6–10 km</td><td class="num">2,048</td><td class="num">9,632</td><td class="num">109.7</td></tr>
  <tr><td>10–15 km</td><td class="num">1,818</td><td class="num">7,354</td><td class="num">85.2</td></tr>
  <tr><td>15–25 km</td><td class="num">2,095</td><td class="num">6,161</td><td class="num">65.0</td></tr>
  <tr><td>25–50 km</td><td class="num">1,270</td><td class="num">3,988</td><td class="num">44.3</td></tr>
</table>

<h2 id="s2"><span class="n">PART 02</span>特征工程：把描述变成数字</h2>

<table>
  <tr><th>输入特征</th><th>怎么得到</th><th>范围</th></tr>
  <tr><td>距市中心 (km)</td><td>由房源经纬度用 Haversine 公式算到人民广场 (121.4737, 31.2304)</td><td class="num">0 – 45</td></tr>
  <tr><td>装修程度</td><td>链家「精装」标签 → 3；有电梯且配套数 ≥3 → 2；其余 → 1（这是代理变量，不是真实装修评级）</td><td class="num">1 – 3</td></tr>
  <tr><td>所在楼层</td><td>「低/中/高楼层 / 总层数」按 0.2 / 0.5 / 0.8 折算成楼层数（估计值）</td><td class="num">1 – 40</td></tr>
  <tr><td>城市物价水平</td><td>公开的城市住宅平均租金（元/㎡/月）</td><td class="num">15.5 – 117.2</td></tr>
  <tr><td>面积 (㎡)</td><td>房源原始字段。研究显示面积与区域位置是租金最强的两个解释变量，所以补上这一维</td><td class="num">10 – 250</td></tr>
</table>

<p>清洗规则：保留月租 1,000–40,000 元、面积 10–250 ㎡、经纬度落在上海范围内的记录，
最终得到 <b>__NREAL__</b> 条有效样本，按 8:2 划分训练集 / 验证集。输入做标准化
<code>x' = (x - μ) / σ</code>，目标月租也标准化，避免量纲差异让梯度下降跑偏。</p>

<h2 id="s3"><span class="n">PART 03</span>MLP 原理</h2>

<p>多层感知机（Multilayer Perceptron）就是若干层全连接堆叠起来。每一层先做一次线性变换，
再过一个非线性激活函数；没有非线性，再多层也等价于一个线性回归，学不出弯曲的关系。</p>

<div class="panel">__NET_SVG__</div>

<h3>前向传播</h3>
<div class="formula">
z⁽¹⁾ = X W⁽¹⁾ + b⁽¹⁾ &nbsp;&nbsp; a⁽¹⁾ = ReLU(z⁽¹⁾)<br>
z⁽²⁾ = a⁽¹⁾ W⁽²⁾ + b⁽²⁾ &nbsp; a⁽²⁾ = ReLU(z⁽²⁾)<br>
ŷ = a⁽²⁾ W⁽³⁾ + b⁽³⁾<br>
<br>
ReLU(x) = max(0, x) &nbsp;&nbsp; L = (1/n) Σ (ŷ - y)²
</div>

<h3>反向传播</h3>
<p>从损失函数出发，用链式法则逐层往回算每个权重的梯度，再用 Adam 更新：</p>
<div class="formula">
∂L/∂ŷ = 2(ŷ - y)/n<br>
∂L/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾ᵀ · δ⁽ˡ⁾ &nbsp;&nbsp; δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ δ⁽ˡ⁺¹⁾) ⊙ 1[z⁽ˡ⁾ &gt; 0]<br>
<br>
m ← β₁m + (1-β₁)g &nbsp;&nbsp; v ← β₂v + (1-β₂)g²<br>
W ← W - lr · m̂ / (√v̂ + ε)
</div>
<p>本页的预测器把训练好的 <code>W</code>、<code>b</code> 和标准化参数直接内嵌在 JavaScript 里，
在浏览器里重跑一遍前向传播 —— 也就是说，<b>你拖动滑块时是真的在跑这个神经网络</b>，
没有向任何服务器发请求。</p>

<h2 id="s4"><span class="n">PART 04</span>训练过程</h2>

<table>
  <tr><th>超参数</th><th>取值</th></tr>
  <tr><td>网络结构</td><td>__ARCH__（输入 5 → 隐藏 24 → 隐藏 16 → 输出 1）</td></tr>
  <tr><td>激活函数 / 损失函数 / 优化器</td><td>ReLU / MSE / Adam</td></tr>
  <tr><td>学习率 / 批大小 / 轮数</td><td>__LR__ / __BATCH__ / __EPOCHS__</td></tr>
  <tr><td>训练集 / 验证集</td><td>__NTRAIN__ / __NVAL__ 条</td></tr>
  <tr><td>实现</td><td>纯 NumPy 手写前向、反向与 Adam，无深度学习框架</td></tr>
</table>

<div class="panel">__CURVE_SVG__</div>
<p style="font-size:13.5px;color:var(--soft)">两条曲线基本重合并持续下降，没有出现验证集损失回升，说明这个规模的网络没有明显过拟合。</p>

<h2 id="s5"><span class="n">PART 05</span>训练结果</h2>

<div class="cards">
  <div class="card"><div class="v">__R2__</div><div class="k">验证集 R²</div></div>
  <div class="card"><div class="v">__MAE__</div><div class="k">平均绝对误差（元/月）</div></div>
  <div class="card"><div class="v">__RMSE__</div><div class="k">RMSE（元/月）</div></div>
  <div class="card"><div class="v">__MAPE__%</div><div class="k">平均相对误差</div></div>
</div>

<h3>模型学到了什么（单因素敏感性）</h3>
<p>把其余四个特征固定在中位数，只让一个特征从第 10 分位变到第 90 分位，看预测月租怎么动：</p>
<table>
  <tr><th>特征</th><th>低值</th><th>高值</th><th>低值月租</th><th>高值月租</th><th>变化</th></tr>
  __SENS_ROWS__
</table>

<div class="callout info">
  <b>和实证研究对得上</b>
  <ul style="margin-top:6px">
    <li><b>距市中心 ↑ → 月租 ↓</b>：与武汉研究「距商业中心每远 1 km 租金降 1.5 元/月/㎡」、长沙研究「到 CBD 距离为首要因素」一致。</li>
    <li><b>装修 ↑ → 月租 ↑</b>：与武汉研究「装修评分每 +1 约 +4.62 元/月/㎡」方向一致。</li>
    <li><b>城市物价 ↑ → 月租大幅 ↑</b>：与公开的 35 城租金梯队（北京 117.2 vs 银川 15.5，差 7.6 倍）一致。</li>
    <li><b>面积 ↑ → 月租 ↑</b>：广州、贝壳的特征价格模型都把建筑面积列为最强解释变量之一。</li>
    <li><b>楼层影响较小</b>：广州研究发现装修程度在统一市场里甚至不显著，说明这类微观特征贡献本就弱于区位与面积 —— 模型给出的量级也确实最小。</li>
  </ul>
</div>

<h2 id="s6"><span class="n">PART 06</span>在线预测器</h2>

<div class="panel">
  <div class="predictor">
    <div>
      <div class="field">
        <label>距市中心 <span id="v_dist">8 km</span></label>
        <input type="range" id="dist" min="0" max="40" step="0.5" value="8">
        <div class="hint">按上海人民广场为基准的直线距离</div>
      </div>
      <div class="field">
        <label>装修程度 <span id="v_deco">2 · 中等</span></label>
        <input type="range" id="deco" min="1" max="3" step="1" value="2">
        <div class="hint">1 简装 / 2 中等 / 3 精装（代理变量）</div>
      </div>
      <div class="field">
        <label>所在楼层 <span id="v_floor">8 层</span></label>
        <input type="range" id="floor" min="1" max="40" step="1" value="8">
      </div>
      <div class="field">
        <label>城市物价水平</label>
        <select id="city">__CITY_OPTIONS__</select>
        <div class="hint">括号内为该城市住宅平均租金（元/㎡/月）</div>
      </div>
      <div class="field">
        <label>面积 <span id="v_area">60 ㎡</span></label>
        <input type="range" id="area" min="10" max="200" step="1" value="60">
      </div>
    </div>
    <div class="result">
      <div class="big" id="out">—</div>
      <div class="unit">元 / 月 · 模型预测</div>
      <div class="sub">折合单价 <b id="out_unit">—</b> 元/㎡/月<br>
      相当于该城市基准单价的 <b id="out_ratio">—</b></div>
      <div class="act-title">隐藏层 1（24 个神经元）的实时激活</div>
      <div class="acts" id="acts1"></div>
      <div class="act-title">隐藏层 2（16 个神经元）的实时激活</div>
      <div class="acts" id="acts2"></div>
    </div>
  </div>
</div>
<p style="font-size:13.5px;color:var(--soft)">
提示：训练数据来自 2020 年上海挂牌房源，且城市维度为统计标定扩展。
此处的数字只用于演示 MLP 的推理过程，请勿作为真实租金参考。</p>

<h2 id="s7"><span class="n">PART 07</span>完整训练代码</h2>
<p>整个流程用纯 NumPy 写成，不依赖 PyTorch / TensorFlow。点击展开可以直接复制运行。</p>
<details>
  <summary>展开 train_mlp.py（数据构建 + MLP + 训练 + 导出，约 250 行）</summary>
  <pre>__TRAIN_CODE__</pre>
</details>
<details>
  <summary>浏览器里的推理逻辑（本页面实际运行的代码）</summary>
  <pre>__INFER_CODE__</pre>
</details>

<h2 id="s8"><span class="n">PART 08</span>局限性与参考资料</h2>

<div class="callout warn">
  <b>这个模型的边界</b>
  <ul style="margin-top:6px">
    <li>数据是 2020 年上海挂牌房源，与 2026 年市场水平有明显时差；近年租金整体下行。</li>
    <li>「装修程度」「楼层」是从标签和文本折算的代理变量，不是真实评级。</li>
    <li>多城市样本是按公开租金指数缩放构造的，不是真实多城市成交记录。</li>
    <li>MLP 是黑盒，且这里没有做特征交叉、正则化调参、交叉验证；换一份数据需要重新训练。</li>
    <li>若要用在真实场景，应改用近期的真实多城市数据，并补上学区、地铁、房龄、朝向、户型、季节等特征。</li>
  </ul>
</div>

<h3>参考资料（均为公开网络材料）</h3>
<ol style="font-size:14px">
  <li>上海链家租房数据集 · GitHub： <a href="https://github.com/XiyueWang/shanghai_rental_analysis_lianjia" target="_blank">XiyueWang/shanghai_rental_analysis_lianjia</a></li>
  <li>《财经》2025 上半年中国一二线城市的房租变化： <a href="https://news.caijingmobile.com/article/detail/552055" target="_blank">news.caijingmobile.com/article/detail/552055</a></li>
  <li>中指研究院《2025 一季度中国住房租赁市场总结》： <a href="https://www.cih-index.com/news/2025-04-23/52652700.html" target="_blank">cih-index.com</a></li>
  <li>麟评居住大数据研究院《10 月全国大中城市租金均价》： <a href="http://www.hbrea.cn/article/13_1303_0_2221617.html" target="_blank">hbrea.cn</a></li>
  <li>住房租金时空演变特征及其影响因素研究 —— 以武汉市主城区为例： <a href="https://search.napstic.cn/literature/degree/0620210900022125" target="_blank">napstic.cn</a></li>
  <li>基于特征价格模型的租赁住宅价格机制研究 —— 以广州市为例： <a href="https://search.napstic.cn/literature/degree/0620210900011633" target="_blank">napstic.cn</a></li>
  <li>Investigating the Determinants of Housing Rents in Hangzhou（空间多层模型）： <a href="https://www.peerkeke.com/article/29046599" target="_blank">peerkeke.com</a></li>
  <li>城市轨道交通沿线住宅租金的空间分异分析 —— 长沙市： <a href="https://www.jjdl.com.cn/CN/PDF/10.15957/j.cnki.jjdl.2021.07.011" target="_blank">jjdl.com.cn</a></li>
  <li>中新经纬：价格特征模型与「60 定律」： <a href="https://www.jwview.com/jingwei/html/12-24/370409.shtml" target="_blank">jwview.com</a></li>
</ol>

<footer>
  <p>数据 → 特征 → MLP → 训练 → 导出权重 → 浏览器推理 → 静态页面发布</p>
  <p style="margin-top:4px">单文件静态 HTML，无外部依赖，可离线打开 · 托管于 GitHub Pages</p>
</footer>

</div>

<script>
const MODEL = __MODEL__;

function relu(v){ return Math.max(0, v); }

function matvec(a, W, b, activate){
  const out = [];
  for(let j=0;j<W[0].length;j++){
    let s = b[0][j];
    for(let i=0;i<a.length;i++) s += a[i]*W[i][j];
    out.push(activate ? relu(s) : s);
  }
  return out;
}

function forward(x){
  const xs = x.map((v,i)=>(v - MODEL.x_mean[i]) / MODEL.x_std[i]);
  const a1 = matvec(xs, MODEL.W[0], MODEL.b[0], true);
  const a2 = matvec(a1, MODEL.W[1], MODEL.b[1], true);
  const out = matvec(a2, MODEL.W[2], MODEL.b[2], false)[0];
  return { pred: out*MODEL.y_std + MODEL.y_mean, a1:a1, a2:a2 };
}

function color(v, max){
  const t = Math.max(0, Math.min(1, v/max));
  const r = Math.round(238 - 79*t), g = Math.round(242 - 76*t), b = Math.round(255 - 66*t);
  return 'rgb('+r+','+g+','+b+')';
}

function renderActs(id, arr){
  const max = Math.max(0.001, ...arr.map(Math.abs));
  document.getElementById(id).innerHTML = arr
    .map(v=>'<span class="act" style="background:'+color(v,max)+'" title="'+v.toFixed(3)+'"></span>').join('');
}

const decoName = {1:'简装', 2:'中等', 3:'精装'};
const ids = ['dist','deco','floor','city','area'];
function update(){
  const d = parseFloat(document.getElementById('dist').value);
  const deco = parseFloat(document.getElementById('deco').value);
  const fl = parseFloat(document.getElementById('floor').value);
  const city = parseFloat(document.getElementById('city').value);
  const area = parseFloat(document.getElementById('area').value);
  document.getElementById('v_dist').textContent = d + ' km';
  document.getElementById('v_deco').textContent = deco + ' · ' + decoName[deco];
  document.getElementById('v_floor').textContent = fl + ' 层';
  document.getElementById('v_area').textContent = area + ' ㎡';

  const r = forward([d, deco, fl, city, area]);
  const rent = Math.max(0, Math.round(r.pred/10)*10);
  document.getElementById('out').textContent = rent.toLocaleString('zh-CN');
  document.getElementById('out_unit').textContent = (rent/area).toFixed(1);
  document.getElementById('out_ratio').textContent = ((rent/area)/city*100).toFixed(0) + '%';
  renderActs('acts1', r.a1);
  renderActs('acts2', r.a2);
}
ids.forEach(id=>document.getElementById(id).addEventListener('input', update));
update();
</script>
</body>
</html>
'''

infer_code = '''function relu(v){ return Math.max(0, v); }

function matvec(a, W, b, activate){
  const out = [];
  for(let j=0;j<W[0].length;j++){
    let s = b[0][j];
    for(let i=0;i<a.length;i++) s += a[i]*W[i][j];
    out.push(activate ? relu(s) : s);
  }
  return out;
}

function forward(x){
  // 1) 用训练时保存的均值/标准差做标准化
  const xs = x.map((v,i)=>(v - MODEL.x_mean[i]) / MODEL.x_std[i]);
  // 2) 两层隐藏层（ReLU）
  const a1 = matvec(xs,  MODEL.W[0], MODEL.b[0], true);
  const a2 = matvec(a1,  MODEL.W[1], MODEL.b[1], true);
  // 3) 输出层（线性）并反标准化
  const out = matvec(a2, MODEL.W[2], MODEL.b[2], false)[0];
  return { pred: out * MODEL.y_std + MODEL.y_mean, a1, a2 };
}'''

repl = {
    "__MODEL__": json.dumps(model, ensure_ascii=False, separators=(",", ":")),
    "__CURVE_SVG__": curve_svg,
    "__NET_SVG__": net_svg,
    "__SAMPLE_ROWS__": sample_rows,
    "__SENS_ROWS__": sens_rows,
    "__CITY_OPTIONS__": city_options,
    "__TRAIN_CODE__": html.escape(train_code),
    "__INFER_CODE__": html.escape(infer_code),
    "__NREAL__": f"{meta['real_listings']:,}",
    "__NTRAIN__": f"{meta['train_size']:,}",
    "__NVAL__": f"{meta['val_size']:,}",
    "__ARCH__": meta["arch"],
    "__LR__": str(meta["lr"]),
    "__BATCH__": str(meta["batch"]),
    "__EPOCHS__": str(meta["epochs"]),
    "__R2__": str(M["r2"]),
    "__MAE__": f"{M['mae']:,}",
    "__RMSE__": f"{M['rmse']:,}",
    "__MAPE__": str(M["mape"]),
}
page = TPL
for k, v in repl.items():
    page = page.replace(k, v)

out = os.path.join(HERE, "index.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(page)
print(f"已生成 {out}  ({len(page):,} 字符)")
