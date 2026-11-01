#!/usr/bin/env python
"""Build the illustrated traffic sign dataset catalog (Japanese HTML and PDF).

The catalog shows, for every representative dataset that is available locally, the definition
of each class or category next to real samples drawn with a fixed seed. Sections are included
when their data is present under ``data/`` (see docs/datasets.md for the download commands):

- GTSRB (43 classes) with Synset Signset Germany synthetic twins
- GTSDB (frames, categories, crops, box sizes)
- Panoramax classified DE road signs (134 classes)
- Synset Signset Germany (modalities and rendering metadata)
- TT100K (category code families, frames, crops)
- Mapillary Traffic Sign Dataset, MTSD (taxonomy, frames, crops)
- CURE-TSR (14 sign types, 12 challenge types x 5 levels, real and unreal)
- Japanese 道路標識令 pictograms (official categories; no public dataset exists)

Usage:
    python scripts/make_dataset_catalog.py --out runs/catalog --pdf
    python scripts/make_dataset_catalog.py --out runs/catalog --only gtsrb,gtsdb

The PDF is rendered with Playwright/Chromium (``pip install playwright`` plus a Chromium build).
All thumbnails are embedded as data URIs so the HTML is self contained.
"""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import html
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from opentsr.signs.datasets.gtsdb import DANGER, MANDATORY, PROHIBITORY, category_of, load_gtsdb_train  # noqa: E402
from opentsr.signs.datasets.gtsrb import SHORT_NAMES  # noqa: E402

# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------


def data_uri(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def fit(img: Image.Image, max_side: int, upscale: bool = True) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1 or upscale:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def thumb_html(img: Image.Image, size: int = 64, title: str = "", cls: str = "th") -> str:
    t = fit(img, size)
    return f'<img class="{cls}" src="{data_uri(t)}" title="{html.escape(title)}" alt="">'


def crop_box(img: Image.Image, box: tuple[float, float, float, float], margin: float = 0.12,
             min_side: int = 8) -> Image.Image:
    """Crop an x1, y1, x2, y2 box with a relative margin, clamped to the image."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = max(w, h, min_side)
    m = side * margin
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = side / 2 + m
    l, t = max(0, int(cx - half)), max(0, int(cy - half))
    r, b = min(img.width, int(cx + half) + 1), min(img.height, int(cy + half) + 1)
    return img.crop((l, t, r, b))


def draw_frame(img: Image.Image, boxes: list[tuple[tuple[float, float, float, float], str, str]],
               width: int = 760) -> Image.Image:
    """Resize a frame to ``width`` px and draw boxes (box, color, label)."""
    s = width / img.width
    out = img.convert("RGB").resize((width, max(1, round(img.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(out)
    for (x1, y1, x2, y2), color, label in boxes:
        bx = [x1 * s, y1 * s, x2 * s, y2 * s]
        d.rectangle(bx, outline=color, width=3)
        if label:
            tw = 5.2 * len(label) + 4
            ty = bx[1] - 10 if bx[1] > 12 else bx[3] + 1
            d.rectangle([bx[0], ty, bx[0] + tw, ty + 10], fill=color)
            d.text((bx[0] + 2, ty - 1), label, fill="white")
    return out


def esc(s: object) -> str:
    return html.escape(str(s))


def size_bucket(side: float) -> str:
    for lim, name in ((16, "<16"), (32, "16-31"), (48, "32-47"), (64, "48-63"), (96, "64-95"), (128, "96-127")):
        if side < lim:
            return name
    return ">=128"


BUCKETS = ["<16", "16-31", "32-47", "48-63", "64-95", "96-127", ">=128"]


def hist_table(values: list[float], caption: str) -> str:
    c = collections.Counter(size_bucket(v) for v in values)
    n = max(1, len(values))
    rows = "".join(
        f"<tr><td>{b} px</td><td class='num'>{c.get(b, 0):,}</td><td class='num'>{100 * c.get(b, 0) / n:.1f}%</td>"
        f"<td><div class='bar' style='width:{140 * c.get(b, 0) / n:.0f}px'></div></td></tr>" for b in BUCKETS
    )
    return (f"<table class='small hist'><caption>{caption}</caption><tr><th>長辺</th><th>個数</th><th>割合</th><th></th></tr>"
            f"{rows}</table>")


# --------------------------------------------------------------------------------------
# Class definitions (Japanese glosses)
# --------------------------------------------------------------------------------------

GTSRB_JA = [
    "速度制限 20 km/h", "速度制限 30 km/h", "速度制限 50 km/h", "速度制限 60 km/h", "速度制限 70 km/h",
    "速度制限 80 km/h", "速度制限 80 km/h 解除", "速度制限 100 km/h", "速度制限 120 km/h", "追越し禁止",
    "3.5 t 超車両の追越し禁止", "次の交差点で優先", "優先道路", "前方優先道路（譲れ）", "止まれ",
    "車両通行止め", "3.5 t 超車両通行止め", "進入禁止", "その他の危険（一般警戒）", "左カーブ注意",
    "右カーブ注意", "連続カーブ注意（最初は左）", "路面凸凹注意", "すべりやすい路面", "右側幅員減少",
    "工事中", "信号機あり", "歩行者注意", "子供注意", "自転車注意", "凍結・積雪注意",
    "野生動物注意", "速度・追越し制限すべて解除", "右折せよ", "左折せよ", "直進のみ", "直進または右折",
    "直進または左折", "右側通行", "左側通行", "環状交差点", "追越し禁止解除", "3.5 t 超車両の追越し禁止解除",
]

GTSDB_CATEGORY_JA = {
    "prohibitory": ("禁止 (prohibitory)", "円形、白地、赤枠。速度制限、追越し禁止、通行止めなど", "#d62728"),
    "mandatory": ("指示 (mandatory)", "円形、青地、白い矢印。進行方向の指定、側方通行、環状交差点", "#1f77b4"),
    "danger": ("警戒 (danger)", "正三角形（頂点が上）、白地、赤枠。カーブ、工事、歩行者などの警告", "#ff7f0e"),
    "other": ("その他 (other)", "上記以外の形状: 止まれ（八角形）、譲れ（逆三角形）、優先道路（菱形）、進入禁止（赤円に白横棒）、解除標識（白円に斜線）", "#2ca02c"),
}

# Panoramax classified_de_road_signs folder names -> Japanese gloss (StVO numbers are indicative).
PANORAMAX_JA = {
    "arrow:red": "赤矢印板（迂回・方向指示の矢印）", "arrow_left_down": "左下向き矢印（補助: 規制範囲の方向）",
    "arrow_right": "右向き矢印（補助）", "arrow_right_down": "右下向き矢印（補助）", "arrow_through": "直進矢印（補助）",
    "arrow_turn_left": "左折矢印（補助）", "arrow_turn_left_right": "左右折矢印（補助）",
    "arrow_turn_left_through": "直進・左折矢印", "arrow_turn_right": "右折矢印（補助）",
    "arrow_turn_right_through": "直進・右折矢印", "arrow_u_turn": "転回矢印",
    "bad": "非標識（検出器の誤検出）", "bad:back": "標識の裏面", "bad:buildings": "建物（誤検出）",
    "bad:round": "円形の非標識物（誤検出）", "bad:windows": "窓（誤検出）",
    "bicycle:priority": "自転車優先道路 Fahrradstraße (244.1)", "bicycle:start": "自転車道 (237)",
    "bus:start": "バス専用車線 (245)", "exit": "出口 Ausfahrt (333)", "give_way": "前方優先道路・譲れ (205)",
    "hazard": "危険箇所一般 (101)", "hazard:airport": "低空飛行機注意 (101-10)", "hazard:animals": "家畜注意 (101-12)",
    "hazard:bicycle": "自転車注意 (138)", "hazard:bothways": "対面通行注意 (125)", "hazard:bumps": "路面凸凹 (112)",
    "hazard:crossing": "交差点注意（右方優先）(102)", "hazard:gravel": "砂利飛散注意 (101-52)",
    "hazard:horse": "乗馬注意 (101-13)", "hazard:ice": "凍結注意 (101-51)", "hazard:incline": "急な坂 (108/110)",
    "hazard:intersection": "交差点注意", "hazard:left_turn": "左カーブ (103-10)", "hazard:level_crossing": "踏切（遮断機あり）(150)",
    "hazard:moving_bridge": "可動橋 (101-55)", "hazard:narrow_both": "幅員減少（両側）(120)",
    "hazard:narrow_left": "幅員減少（左）(121-20)", "hazard:narrow_right": "幅員減少（右）(121-10)",
    "hazard:other": "その他の危険（補助文字付き）", "hazard:pedestrian_crossing": "横断歩道注意 (101-11)",
    "hazard:right_turn": "右カーブ (103-20)", "hazard:road_works": "工事中 (123)", "hazard:rocks": "落石注意 (101-15)",
    "hazard:school": "子供注意 (136)", "hazard:slippery": "すべりやすい路面 (114)", "hazard:traffic_jam": "渋滞注意 (124)",
    "hazard:traffic_signal": "信号機あり (131)", "hazard:train": "踏切予告（遮断機なし）(151)", "hazard:train2": "列車注意（踏切予告 2）",
    "hazard:wild_animals": "野生動物注意 (142)", "hazard:wind": "横風注意 (117)", "hazard:zigzag": "連続カーブ（最初は右）(105-20)",
    "hazard:zigzag:left": "連続カーブ（最初は左）(105-10)", "horse:start": "乗馬道 (238)", "lanes": "車線表示板",
    "maxaxleweight": "軸重制限 (263)", "maxheight": "高さ制限 (265)", "maxlength": "長さ制限 (266)",
    "maxspeed:5": "最高速度 5 km/h (274)", "maxspeed:10": "最高速度 10 km/h (274)", "maxspeed:20": "最高速度 20 km/h (274)",
    "maxspeed:25": "最高速度 25 km/h (274)", "maxspeed:30": "最高速度 30 km/h (274)", "maxspeed:40": "最高速度 40 km/h (274)",
    "maxspeed:50": "最高速度 50 km/h (274)", "maxspeed:60": "最高速度 60 km/h (274)", "maxspeed:70": "最高速度 70 km/h (274)",
    "maxspeed:80": "最高速度 80 km/h (274)", "maxspeed:100": "最高速度 100 km/h (274)", "maxspeed:120": "最高速度 120 km/h (274)",
    "maxspeed:130": "最高速度 130 km/h (274)", "maxspeed:end": "最高速度制限解除 (278)", "maxweight": "重量制限 (262)",
    "maxwidth": "幅制限 (264)", "min_distance": "最小車間距離 (273)", "motorway:end": "高速道路終わり (330.2)",
    "motorway:start": "高速道路 Autobahn (330.1)", "no:end": "すべての区間規制解除 (282)", "no:other": "その他の禁止",
    "no:other:dog": "犬の立入禁止", "no_bicycle": "自転車通行止め (254)", "no_bus": "バス通行止め (257-54)",
    "no_car": "自動車通行止め (251)", "no_car_bicycle": "自動車・自転車通行止め", "no_car_motorcycle": "自動車・二輪車通行止め (260)",
    "no_dangerous": "危険物積載車両通行止め (261)", "no_dangerous_liquid": "水質汚染物質積載車両通行止め (269)",
    "no_entry": "進入禁止 (267)", "no_exit": "行き止まり (357)", "no_exit:car": "車両行き止まり（歩行者・自転車可）",
    "no_hgv": "貨物自動車通行止め (253)", "no_horse": "乗馬禁止 (257-51)", "no_motorcycle": "二輪車通行止め (255)",
    "no_overtaking": "追越し禁止 (276)", "no_overtaking:end": "追越し禁止解除 (280)", "no_overtaking:hgv": "貨物車追越し禁止 (277)",
    "no_parking": "駐車禁止 (286)", "no_pedestrian": "歩行者通行止め (259)", "no_stop": "停車禁止 (283)",
    "no_u_turn": "転回禁止 (272)", "no_way": "全車両通行止め (250)", "oneway": "一方通行 (220)", "other": "その他の標識",
    "other:charging": "充電スタンド案内", "other:motorhome": "キャンピングカー関連", "other:multi": "複合標識板",
    "park_ride": "パーク&ライド (316)", "parking": "駐車場 (314)", "parking:covered": "屋内駐車場", "parking:disabled": "障害者用駐車",
    "parking:pedestrian": "歩道上駐車 (315)", "parking:perpendicular": "直角駐車 (315)", "parking:sidewalk": "歩道縁石駐車 (315)",
    "pedestrian:start": "歩行者専用 (239)", "pedestrian_bicycle:designated": "歩行者・自転車共用道 (240)",
    "pedestrian_bicycle:start": "歩行者・自転車分離道 (241)", "pedestrian_crossing": "横断歩道 (350)",
    "priority:backward": "対向車優先 (208)", "priority:end": "優先道路終わり (307)", "priority:forward": "対向車に対し優先 (308)",
    "priority:start": "優先道路 (306)", "roundabout": "環状交差点 (215)", "stop": "止まれ (206)", "traffic_signal": "信号機（物体）",
    "trunk:end": "自動車専用道路終わり (331.2)", "trunk:start": "自動車専用道路 (331.1)", "tunnel:start": "トンネル (327)",
    "zone:20": "ゾーン 20 (274.1)", "zone:30": "ゾーン 30 (274.1)", "zone:30:end": "ゾーン 30 終わり (274.2)",
    "zone:end": "ゾーン終わり", "zone:no_parking": "駐車禁止ゾーン (290.1)", "zone:pedestrian": "歩行者ゾーン (242.1)",
}

PANORAMAX_FAMILIES = [
    ("hazard:*", "警戒標識（三角形、赤枠）", "Gefahrzeichen 101 から 151"),
    ("maxspeed:*, zone:*", "最高速度・ゾーン規制", "274, 274.1/274.2"),
    ("no_*, no:*", "通行止め・禁止（円形、赤枠）", "250 から 286"),
    ("max*", "重量・高さ・幅・長さ・軸重の制限", "262 から 266"),
    ("priority:*, give_way, stop, roundabout", "優先関係", "205, 206, 208, 215, 306 から 308"),
    ("*:start, *:end", "専用道・道路種別の開始と終了", "237 から 245, 330, 331"),
    ("parking*, park_ride", "駐車関連（青地、白 P）", "314 から 316"),
    ("arrow*", "補助標識の矢印板", "Zusatzzeichen 1000 番台"),
    ("bad*, other*", "非標識（誤検出の負例）とその他", "分類器用の補助クラス"),
]

TT100K_FAMILIES = [
    ("p*", "禁令標志（禁止）: 円形、白地、赤枠", "p1 から p29 は個別の禁止事項、po はその他の禁止"),
    ("pl*", "限速（最高速度）: 数字が制限値 km/h", "pl5 から pl120"),
    ("pr*", "解除限速（速度制限解除）", "pr10 から pr100"),
    ("pm*", "限制质量（重量制限）: 数字は t", "pm1.5 から pm55"),
    ("ph*", "限制高度（高さ制限）: 数字は m", "ph1.8 から ph5.5"),
    ("pw*", "限制宽度（幅制限）: 数字は m", "pw2.5 から pw4.5"),
    ("pa*", "限制轴重（軸重制限）: 数字は t", "pa6 から pa18"),
    ("pn, pne, pnl, pnlc", "禁止停车・禁止驶入・禁止长时停车 など", "pn 駐車禁止, pne 進入禁止, pnl 長時間駐車禁止"),
    ("ps, pg, pt, pe, pd, pb, pc*, pr*", "停车让行（止まれ）、减速让行（譲れ）など", "ps 止まれ, pg 譲れ"),
    ("i*", "指示标志（指示）: 円形または四角、青地", "i1 から i15, io はその他の指示"),
    ("il*", "最低限速（最低速度）", "il50 から il110"),
    ("ip, iz, im", "人行横道（横断歩道）など", "ip 横断歩道"),
    ("w*", "警告标志（警戒）: 三角形、黄地、黒枠", "w1 から w67, wo はその他の警戒"),
]

# Meanings of the 45 TT100K benchmark classes (those with more than 100 instances in the 2016 release).
TT100K_JA = {
    "i2": "非機動車（自転車）通行", "i4": "機動車（自動車）通行", "i5": "右側通行", "il100": "最低速度 100 km/h",
    "il60": "最低速度 60 km/h", "il80": "最低速度 80 km/h", "io": "その他の指示標識", "ip": "横断歩道",
    "p10": "機動車進入禁止", "p11": "警笛禁止", "p12": "二輪車進入禁止", "p19": "右折禁止", "p23": "左折禁止",
    "p26": "貨物車進入禁止", "p27": "危険物積載車進入禁止", "p3": "大型バス進入禁止", "p5": "転回禁止",
    "p6": "非機動車進入禁止", "pg": "減速譲れ", "ph4": "高さ制限 4 m", "ph4.5": "高さ制限 4.5 m", "ph5": "高さ制限 5 m",
    "pl100": "最高速度 100 km/h", "pl120": "最高速度 120 km/h", "pl20": "最高速度 20 km/h", "pl30": "最高速度 30 km/h",
    "pl40": "最高速度 40 km/h", "pl5": "最高速度 5 km/h", "pl50": "最高速度 50 km/h", "pl60": "最高速度 60 km/h",
    "pl70": "最高速度 70 km/h", "pl80": "最高速度 80 km/h", "pm20": "重量制限 20 t", "pm30": "重量制限 30 t",
    "pm55": "重量制限 55 t", "pn": "駐車禁止", "pne": "進入禁止", "po": "その他の禁止標識", "pr40": "速度制限 40 解除",
    "w13": "十字交差点", "w32": "工事中", "w55": "子供注意", "w57": "歩行者注意", "w59": "合流注意", "wo": "その他の警戒標識",
    "ps": "止まれ", "pl10": "最高速度 10 km/h", "pl15": "最高速度 15 km/h", "pl25": "最高速度 25 km/h",
    "pl35": "最高速度 35 km/h", "pl90": "最高速度 90 km/h", "pl110": "最高速度 110 km/h", "il50": "最低速度 50 km/h",
    "il70": "最低速度 70 km/h", "il90": "最低速度 90 km/h", "il110": "最低速度 110 km/h", "pr20": "速度制限 20 解除",
    "pr30": "速度制限 30 解除", "pr50": "速度制限 50 解除", "pr60": "速度制限 60 解除", "pr70": "速度制限 70 解除",
    "pr80": "速度制限 80 解除", "pr100": "速度制限 100 解除",
}

MTSD_GROUPS = [
    ("regulatory", "規制 (regulatory)", "禁止・制限・指示・優先関係。円形の赤枠や青地、止まれ、譲れなど"),
    ("warning", "警戒 (warning)", "危険の警告。三角形（欧州）や菱形（米州・アジア）の黄地・黒図柄"),
    ("information", "案内 (information)", "施設・方向・駐車・サービスなどの情報提供。青や緑の四角"),
    ("complementary", "補助 (complementary)", "本標識に付く補助板: 距離、方向矢印、対象車種、時間帯、速度の補足"),
    ("other-sign", "その他 (other-sign)", "分類名の付かない標識。検出のみの対象（タクソノミー外）"),
]

MTSD_PROPS = {
    "occluded": "一部が遮蔽されている", "out-of-frame": "画像の端で切れている", "ambiguous": "ラベルが曖昧",
    "exterior": "複数標識を囲む外枠", "included": "他の標識の外枠に含まれる", "direction-or-information": "方向・情報の案内板",
    "highway": "高速道路の標識", "dummy": "ダミー（学習に使わない）", "barrier": "遮断機など標識でないもの",
}

CURE_SIGNS = ["speed_limit", "goods_vehicles", "no_overtaking", "no_stopping", "no_parking", "stop", "bicycle",
              "hump", "no_left", "no_right", "priority_to", "no_entry", "yield", "parking"]
CURE_SIGNS_JA = ["速度制限（すべての値）", "貨物車通行止め", "追越し禁止", "停車禁止", "駐車禁止", "止まれ", "自転車",
                 "路面隆起（ハンプ）", "左折禁止", "右折禁止", "対向車に対し優先", "進入禁止", "譲れ", "駐車場"]
CURE_CHALLENGES = ["challenge free", "decolorization", "lens blur", "codec error", "darkening", "dirty lens", "exposure",
                   "gaussian blur", "noise", "rain", "shadow", "snow", "haze"]
CURE_CHALLENGES_JA = ["劣化なし", "脱色", "レンズぼけ", "コーデック誤り", "暗化", "レンズ汚れ", "露出過多",
                      "ガウスぼけ", "ノイズ", "雨", "影", "雪", "霧"]

OVERVIEW = [
    # name, country, task, scale, classes, license, used here
    ("GTSRB (2011)", "ドイツ", "分類（切り抜き）", "51,839 枚（学習 39,209 / テスト 12,630）", "43",
     "\"free to use\" + 引用依頼（正式なライセンス名なし）", "公式分割の全体（実例はテスト）"),
    ("GTSDB (2013)", "ドイツ", "検出（フレーム）", "900 枚 1360x800、1,206 個（学習 gt.txt は 852 個）", "43 / 4 カテゴリ",
     "\"free to use for any purpose\"（ReadMe）", "学習 600 枚 + gt.txt"),
    ("Panoramax classified DE (2024-26)", "ドイツ（＋BE/FR 補充）", "分類（切り抜き）", "20,000 枚超", "134",
     "CC BY-SA 4.0", "validation 6,944 枚"),
    ("Synset Signset Germany (2024)", "ドイツ（合成）", "分類（合成、マスク付き）", "105,500 枚（43 クラス版 21,500）", "211 / 43",
     "CC BY 4.0", "Cycles 版 validation 4,300 枚"),
    ("TT100K (2016/2021)", "中国", "検出＋分類", "100,000 枚 2048x2048、約 30,000 個", "221 / 232（45 が標準）",
     "CC BY-NC（商用は要相談）", "ミラー 3 シャード"),
    ("MTSD (2020)", "世界", "検出＋分類", "約 100,000 枚、257,543 個（完全注記 52,453 枚）", "313 / 400 + other",
     "CC BY-NC-SA 4.0", "公式注記付き画像の標本（約 300 枚）"),
    ("CURE-TSR (2017)", "ベルギー実映像＋合成", "分類（頑健性）", "2,206,106 枚", "14（12 劣化 x 5 レベル）",
     "CC BY 4.0（Zenodo 版）", "Zenodo zip 全体のファイル名解析 + 実例"),
    ("日本 道路標識令", "日本", "（公開データセットなし）", "163 種類 / 237 図柄（法令別表）", "本標識 4 分類 + 補助",
     "法令由来の図柄（著作権法第 13 条）", "国土交通省「道路標識一覧」の公式図柄 221 点"),
]

NOT_INCLUDED = [
    ("ZOD (Zenseact Open Dataset) Frames", "欧州 14 か国", "156 標識クラス / 44.6 万個", "CC BY-SA 4.0", "devkit 経由の登録制ダウンロード（クラウドから取得不可）。商用可の大規模細分類データとして次版の最優先候補"),
    ("DFG Traffic Sign Dataset", "スロベニア", "200 / 6,957 枚 13,239 個", "CC BY-NC-SA 4.0", "非商用、公式サイト直接配布のみ"),
    ("Swedish Traffic Signs (STS)", "スウェーデン", "7 / 2 万枚超", "Zenodo 版 CC BY 4.0", "15.8 GB の一括アーカイブで本セッションでは取得せず"),
    ("BelgiumTS (BTSC / BTSD)", "ベルギー", "62 分類 / 108 検出", "引用依頼のみ", "CURE-TSR の実画像が BelgiumTS 由来のため重複。次版で切り抜きを追加予定"),
    ("LISA Traffic Sign", "米国", "47 / 6,610 枚 7,855 個", "academic license agreement", "ライセンス上サンプル掲載を避けた"),
    ("ATSD (Austrian Highway)", "オーストリア", "108 / 7,454 枚", "\"Permissive license\"（条件 PDF 未確認）", "条件未確認"),
    ("RTSD (Russian)", "ロシア", "205 / 約 6 万枚", "記載なし", "Yandex Disk 配布でクラウドから取得不可"),
    ("CCTSDB 2021", "中国", "3 / 17,856 枚", "GPL-3.0", "3 クラス（禁止・指示・警告）のみで細分類がない"),
    ("ETSD (European)", "欧州 6 か国", "164 / 8 万枚超", "学術目的、要申請", "申請制"),
    ("English roadsigns", "英国", "31 / 55,000 切り抜き", "CC BY 4.0", "本版では未取得。再配布可能な切り抜きデータとして次版候補"),
    ("Panoramax classified FR", "フランス", "250 超 / 66,000 切り抜き", "etalab-2.0", "DE 版で Panoramax の性質は示せるため省略"),
]

JP_CATEGORY_DEF = [
    ("案内標識", "地点・方面・距離・路線番号・施設などを案内する。青地（一般道）または緑地（高速道路）の四角形。番号 101 から 120 番台"),
    ("警戒標識", "道路上の危険や注意すべき状況を予告する。黄地に黒図柄の菱形（ひし形）。番号 201 から 215"),
    ("規制標識", "禁止・制限・指定を示す。円形（赤枠・青地）、逆三角形（徐行・一時停止）など。番号 301 から 332"),
    ("指示標識", "通行方法や施設を指示する。青地の四角形または円形。番号 401 から 415"),
    ("補助標識", "本標識の意味を補足する白地の板（距離・区域、日・時間、車両の種類、始まり・終わりなど）。番号 501 から 516"),
]

# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


class Catalog:
    def __init__(self, seed: int, data_root: Path):
        self.seed = seed
        self.data = data_root
        self.parts: list[str] = []
        self.toc: list[tuple[str, str, str]] = []  # (anchor, title, summary)
        self.sources: list[tuple[str, str, str, str]] = []  # dataset, local files, license, url
        self.n_images = 0

    def rng(self, name: str) -> random.Random:
        return random.Random(f"{self.seed}:{name}")

    def section(self, anchor: str, title: str, summary: str, body: str) -> None:
        self.toc.append((anchor, title, summary))
        self.parts.append(f'<section class="ds" id="{anchor}"><h2>{title}</h2>{body}</section>')

    # ---------------------------------------------------------------- GTSRB + Synset
    def gtsrb(self, n_real: int = 6, n_syn: int = 2) -> None:
        root = self.data / "gtsrb_hf"
        syn = self.data / "synset_signset" / "cycles" / "validation.parquet"
        if not (root / "data").exists():
            print("skip gtsrb: no data")
            return
        rng = self.rng("gtsrb")

        def read(split: str):
            path = sorted((root / "data").glob(f"{split}-*.parquet"))[0]
            return pq.read_table(path)

        train = read("train")
        test = read("test")
        tr_labels = np.asarray(train.column("ClassId").to_pylist())
        te_labels = np.asarray(test.column("ClassId").to_pylist())
        tr_w = np.asarray(train.column("Width").to_pylist())
        tr_h = np.asarray(train.column("Height").to_pylist())
        te_imgs = test.column("Path").to_pylist()
        syn_by_class: dict[int, list[tuple[bytes, bytes]]] = collections.defaultdict(list)
        syn_n = 0
        if syn.exists():
            t = pq.read_table(syn, columns=["label", "image", "mask"])
            for lab, im, mk in zip(t.column("label").to_pylist(), t.column("image").to_pylist(), t.column("mask").to_pylist()):
                syn_by_class[lab].append((im["bytes"], mk["bytes"]))
            syn_n = t.num_rows

        def syn_crop(im_b: bytes, mk_b: bytes) -> Image.Image:
            im = Image.open(io.BytesIO(im_b)).convert("RGB")
            bbox = Image.open(io.BytesIO(mk_b)).convert("L").point(lambda v: 255 if v > 64 else 0).getbbox()
            if not bbox:
                return im
            return crop_box(im, bbox, margin=0.12)

        rows = []
        for c in range(43):
            tr_idx = np.flatnonzero(tr_labels == c)
            te_idx = np.flatnonzero(te_labels == c)
            pick = rng.sample(list(te_idx), min(n_real, len(te_idx)))
            thumbs = "".join(
                thumb_html(Image.open(io.BytesIO(te_imgs[i]["bytes"])), 56, f"test #{i}") for i in pick)
            self.n_images += len(pick)
            syn_html = ""
            if syn_by_class.get(c):
                sp = rng.sample(syn_by_class[c], min(n_syn, len(syn_by_class[c])))
                syn_html = "".join(thumb_html(syn_crop(b, m), 56, "Synset Signset (synthetic)", "th syn") for b, m in sp)
                self.n_images += len(sp)
            cat = category_of(c)
            med = int(np.median(np.maximum(tr_w[tr_idx], tr_h[tr_idx]))) if len(tr_idx) else 0
            rows.append(
                f"<tr><td class='num'>{c}</td><td><b>{esc(GTSRB_JA[c])}</b><br><span class='en'>{esc(SHORT_NAMES[c])}</span></td>"
                f"<td><span class='tag' style='background:{GTSDB_CATEGORY_JA[cat][2]}'>{cat}</span></td>"
                f"<td class='num'>{len(tr_idx):,}<br>{len(te_idx):,}</td><td class='num'>{med}</td>"
                f"<td class='thumbs'>{thumbs}</td><td class='thumbs'>{syn_html}</td></tr>")

        cat_rows = "".join(
            f"<tr><td><span class='tag' style='background:{v[2]}'>{k}</span></td><td>{esc(v[0])}</td><td>{esc(v[1])}</td>"
            f"<td>{', '.join(str(i) for i in sorted(s))}</td></tr>"
            for k, v, s in (("prohibitory", GTSDB_CATEGORY_JA["prohibitory"], PROHIBITORY),
                            ("mandatory", GTSDB_CATEGORY_JA["mandatory"], MANDATORY),
                            ("danger", GTSDB_CATEGORY_JA["danger"], DANGER),
                            ("other", GTSDB_CATEGORY_JA["other"], set(range(43)) - PROHIBITORY - MANDATORY - DANGER)))
        sizes = np.maximum(tr_w, tr_h)
        body = f"""
<p class="lead">GTSRB（German Traffic Sign Recognition Benchmark, IJCNN 2011）はドイツの実走行映像から切り出した標識画像の分類ベンチマークです。
公式分割は学習 {len(tr_labels):,} 枚、テスト {len(te_labels):,} 枚、43 クラス。1 つの標識を約 1 秒追跡した 30 枚が 1 トラックを構成し、
テストは学習と別のトラックから採られています。画像は標識の周囲に約 10% の余白を含み、サイズは {int(sizes.min())} から {int(sizes.max())} px
（長辺の中央値 {int(np.median(sizes))} px）、ROI 列で標識の矩形が与えられます。クラス id、名称、GTSDB の 4 カテゴリ区分、学習/テスト枚数、
学習画像の長辺中央値、テスト画像から seed 固定で抽出した実例 {n_real} 枚、そして同じクラスの Synset Signset Germany（Fraunhofer IOSB の
合成データ、CC BY 4.0、validation {syn_n:,} 枚から抽出）の合成例 {n_syn} 枚を並べています。</p>
<table class="small"><caption>表: GTSDB 競技会のカテゴリ定義（GTSRB のクラス id を 4 区分に束ねる）</caption>
<tr><th>カテゴリ</th><th>名称</th><th>定義（形状・色）</th><th>該当クラス id</th></tr>{cat_rows}</table>
{hist_table(list(sizes), "表: GTSRB 学習画像の長辺サイズ分布")}
<table class="cls"><caption>表: GTSRB の 43 クラス。実例はテスト分割（seed {self.seed}）、右端は Synset Signset の合成例</caption>
<thead><tr><th>id</th><th>名称（日 / 英）</th><th>区分</th><th>学習<br>テスト</th><th class='nw'>中央値<br>(px)</th><th>実例（GTSRB テスト）</th><th>合成例（Synset）</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">読み方: 速度制限の 8 クラス（0 から 8、6 を除く）は数字だけが違い、カーブ・幅員減少・追越し禁止の各組も図柄の細部だけが違います。
テスト画像には夜間・ブレ・低コントラストのトラックが含まれ、これが CLIP 等の zero-shot モデルが崩れる原因です（技術報告 第 5 節）。</p>
"""
        self.section("gtsrb", "1. GTSRB: 43 クラスの定義と実例（＋ Synset Signset 合成例）",
                     "ドイツ、分類、51,839 枚、43 クラス", body)
        self.sources.append(("GTSRB", "data/gtsrb_hf (HF bazyl/GTSRB, 公式分割)",
                             "公式: \"The data is free to use\" + 引用依頼", "https://benchmark.ini.rub.de/gtsrb_dataset.html"))
        if syn.exists():
            self.sources.append(("Synset Signset Germany (GTSRB subset, Cycles validation)",
                                 "data/synset_signset/cycles/validation.parquet",
                                 "CC BY 4.0（Fraunhofer IOSB の帰属表示依頼あり）",
                                 "https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset"))

    # ---------------------------------------------------------------- GTSDB
    def gtsdb(self, n_frames: int = 6, n_crops: int = 14) -> None:
        root = self.data / "gtsdb_hf"
        if not root.exists():
            print("skip gtsdb: no data")
            return
        rng = self.rng("gtsdb")
        images = load_gtsdb_train(root)
        with_boxes = [im for im in images if im.boxes]
        n_boxes = sum(len(im.boxes) for im in images)
        frames = rng.sample(with_boxes, n_frames)
        frame_html = []
        for im in frames:
            pil = Image.open(im.path)
            boxes = []
            for (x1, y1, x2, y2), c in zip(im.boxes, im.class_ids):
                color = GTSDB_CATEGORY_JA[category_of(c)][2]
                boxes.append(((x1, y1, x2, y2), color, str(c)))
            fr = draw_frame(pil, boxes, 820)
            cap = "、".join(f"{c}: {GTSRB_JA[c]} ({x2 - x1 + 1}x{y2 - y1 + 1} px)" for (x1, y1, x2, y2), c in zip(im.boxes, im.class_ids))
            frame_html.append(f'<figure class="frame"><img src="{data_uri(fr, quality=80)}" alt=""><figcaption>{esc(im.image_id)}.ppm, 1360x800: {esc(cap)}</figcaption></figure>')
            self.n_images += 1
        # crops per category
        per_cat: dict[str, list[tuple]] = collections.defaultdict(list)
        sides = []
        cat_counts: collections.Counter = collections.Counter()
        cls_counts: collections.Counter = collections.Counter()
        for im in images:
            for box, c in zip(im.boxes, im.class_ids):
                per_cat[category_of(c)].append((im, box, c))
                sides.append(max(box[2] - box[0] + 1, box[3] - box[1] + 1))
                cat_counts[category_of(c)] += 1
                cls_counts[c] += 1
        crop_html = []
        for cat in ("prohibitory", "mandatory", "danger", "other"):
            items = rng.sample(per_cat[cat], min(n_crops, len(per_cat[cat])))
            cards = []
            for im, box, c in items:
                pil = Image.open(im.path)
                cr = crop_box(pil, (box[0], box[1], box[2] + 1, box[3] + 1), margin=0.15)
                cards.append(f'<div class="card"><img src="{data_uri(fit(cr, 72))}" alt=""><div class="cap">{c} {esc(GTSRB_JA[c])}<br><span class="en">{box[2] - box[0] + 1}x{box[3] - box[1] + 1} px</span></div></div>')
                self.n_images += 1
            name, defi, color = GTSDB_CATEGORY_JA[cat]
            crop_html.append(f'<h3><span class="tag" style="background:{color}">{cat}</span> {esc(name)}: {esc(defi)}。gt.txt 内 {cat_counts[cat]} 個</h3><div class="grid">{"".join(cards)}</div>')
        items = sorted(cls_counts.items())
        halves = [items[: (len(items) + 1) // 2], items[(len(items) + 1) // 2:]]
        cls_rows = "".join(
            "<table class='small half'><tr><th>id</th><th>名称</th><th>区分</th><th>個数</th></tr>"
            + "".join(f"<tr><td class='num'>{c}</td><td>{esc(GTSRB_JA[c])}</td><td>{category_of(c)}</td><td class='num'>{n}</td></tr>" for c, n in h)
            + "</table>" for h in halves)
        body = f"""
<p class="lead">GTSDB（German Traffic Sign Detection Benchmark, IJCNN 2013）は 1360x800 px の走行フレーム 900 枚（学習 600、評価 300）に
標識の矩形と GTSRB と同じ 43 クラス id を付けた検出ベンチマークです。公開ミラーには学習 600 枚の gt.txt（{n_boxes} 個の矩形、標識を含むフレーム {len(with_boxes)} 枚、
標識なし {len(images) - len(with_boxes)} 枚）と評価 300 枚の画像（正解なし）が含まれます。競技会は 43 クラスではなく上の 3 カテゴリ（禁止・指示・警戒）ごとに検出器を評価しました。
標識の長辺は 16 から 128 px で、半数以上が 48 px 未満です。</p>
<h3>フレームの実例（seed {self.seed} で {n_frames} 枚、枠の色はカテゴリ、数字はクラス id）</h3>
<div class="frames">{''.join(frame_html)}</div>
{hist_table(sides, "表: GTSDB 学習分割 852 個の標識の長辺サイズ分布")}
<h3>カテゴリ別の切り抜き実例（枠の 15% 余白付き）</h3>
{''.join(crop_html)}
<p class="capt">表: GTSDB 学習分割に現れるクラスと個数</p><div class="side">{cls_rows}</div>
"""
        self.section("gtsdb", "2. GTSDB: 検出フレーム、カテゴリ定義、切り抜き", "ドイツ、検出、900 フレーム、43 クラス / 4 カテゴリ", body)
        self.sources.append(("GTSDB", "data/gtsdb_hf (HF GlowBond/GTSDB: TrainIJCNN2013 + Test images)",
                             "ReadMe: \"free to use this data package for any purpose you like\"",
                             "https://benchmark.ini.rub.de/gtsdb_dataset.html"))

    # ---------------------------------------------------------------- Panoramax DE
    def panoramax(self, n: int = 6) -> None:
        root = self.data / "panoramax_de" / "val"
        if not root.exists():
            print("skip panoramax: no data")
            return
        rng = self.rng("panoramax")
        classes = sorted(p for p in root.iterdir() if p.is_dir())
        total = 0
        rows = []
        sizes = []
        for p in classes:
            files = sorted(x for x in p.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
            total += len(files)
            pick = rng.sample(files, min(n, len(files)))
            th = []
            for f in pick:
                im = Image.open(f)
                sizes.append(max(im.size))
                th.append(thumb_html(im, 60, f.name))
                self.n_images += 1
            prefixes = collections.Counter(f.name.split("_")[0] for f in files if "_" in f.name and f.name.split("_")[0].isupper())
            pref = ", ".join(f"{k} {v}" for k, v in prefixes.most_common(3) if k != "DE")
            rows.append(f"<tr><td><code>{esc(p.name)}</code></td><td>{esc(PANORAMAX_JA.get(p.name, ''))}</td>"
                        f"<td class='num'>{len(files)}</td><td class='en'>{esc(pref)}</td><td class='thumbs'>{''.join(th)}</td></tr>")
        fam = "".join(f"<tr><td><code>{esc(a)}</code></td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in PANORAMAX_FAMILIES)
        body = f"""
<p class="lead">Panoramax は自治体・団体・市民が投稿するフランス発のオープンな路上写真基盤で、その標識検出器で切り出した画像を人手で分類した
学習用データセットが Hugging Face で公開されています。ドイツ版 <code>Panoramax/classified_de_road_signs</code>（CC BY-SA 4.0）は
{len(classes)} クラス、ここで用いた validation 分割は {total:,} 枚の切り抜きです。クラス名は OpenStreetMap の <code>traffic_sign</code> タグに近い
意味的な名前で、StVO（ドイツ道路交通規則）の標識番号ではありません。一部のクラスは BE_, FR_ などの接頭辞で他国の標識が補充されています。
負例クラス（bad:* ）が入っている点が、GTSRB のような純粋なベンチマークとの違いです。切り抜きの長辺は中央値 {int(np.median(sizes))} px です。
日本語訳と StVO 番号は本カタログ作成者による参考訳です。</p>
<table class="small"><caption>表: クラス名の接頭辞と意味のまとまり</caption><tr><th>クラス名</th><th>意味</th><th>対応する StVO 標識</th></tr>{fam}</table>
<table class="cls"><caption>表: Panoramax DE の全 {len(classes)} クラス。validation の枚数、他国補充の内訳（ファイル名接頭辞）、実例 {n} 枚（seed {self.seed}）</caption>
<thead><tr><th>クラス</th><th>意味（参考訳、StVO 番号）</th><th>枚数</th><th>他国<br>補充</th><th>実例</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
"""
        self.section("panoramax", f"3. Panoramax classified DE road signs: {len(classes)} クラスの定義と実例",
                     f"ドイツ（＋欧州補充）、分類、validation {total:,} 枚、{len(classes)} クラス", body)
        self.sources.append(("Panoramax classified_de_road_signs (validation)", "data/panoramax_de/val",
                             "CC BY-SA 4.0", "https://huggingface.co/datasets/Panoramax/classified_de_road_signs"))

    # ---------------------------------------------------------------- Synset Signset
    def synset(self, n_examples: int = 4) -> None:
        path = self.data / "synset_signset" / "cycles" / "validation.parquet"
        if not path.exists():
            print("skip synset: no data")
            return
        rng = self.rng("synset")
        t = pq.read_table(path)
        n = t.num_rows
        cols = {c: t.column(c).to_pylist() for c in t.column_names if c not in ("image", "mask", "segmentation")}
        pick = rng.sample(range(n), n_examples)
        imgs = t.column("image").to_pylist()
        masks = t.column("mask").to_pylist()
        segs = t.column("segmentation").to_pylist()
        ex = []
        for i in pick:
            im = Image.open(io.BytesIO(imgs[i]["bytes"]))
            mk = Image.open(io.BytesIO(masks[i]["bytes"]))
            sg = Image.open(io.BytesIO(segs[i]["bytes"]))
            meta = (f"class {cols['label'][i]} {esc(cols['class_name'][i])} / {esc(cols['environment_name'][i])} / "
                    f"{esc(cols['environment_day_time'][i])}, {esc(cols['environment_weather'][i])}, {esc(cols['environment_season'][i])} / "
                    f"motion blur {cols['motion_blur_pixels'][i]:.1f} px, noise {cols['relative_noise_level'][i]:.2f}, upper: {esc(cols['upper_sign'][i])}, lower: {esc(cols['lower_sign'][i])}")
            ex.append(f'<div class="triple"><img src="{data_uri(fit(im, 120))}" alt=""><img src="{data_uri(fit(mk.convert("L"), 120), "PNG")}" alt="">'
                      f'<img src="{data_uri(fit(sg.convert("RGB"), 120))}" alt=""><div class="cap">{meta}</div></div>')
            self.n_images += 3

        def dist(col: str, top: int = 8) -> str:
            c = collections.Counter(str(v) for v in cols[col])
            return ", ".join(f"{esc(k)} {100 * v / n:.0f}%" for k, v in c.most_common(top))

        def rng_of(col: str) -> str:
            v = np.asarray([x for x in cols[col] if x is not None], dtype=float)
            return f"{v.min():.2f} から {v.max():.2f}（中央値 {np.median(v):.2f}）"

        meta_rows = "".join(f"<tr><td><code>{c}</code></td><td>{esc(d)}</td><td>{v}</td></tr>" for c, d, v in [
            ("rendering_engine", "レンダラ", dist("rendering_engine")),
            ("environment_day_time", "時間帯", dist("environment_day_time")),
            ("environment_weather", "天候", dist("environment_weather")),
            ("environment_season", "季節", dist("environment_season")),
            ("environment_location", "場所の種類", dist("environment_location")),
            ("environment_contrast / light", "コントラスト / 光", dist("environment_contrast", 4) + " / " + dist("environment_light", 4)),
            ("bayer_pattern", "ベイヤー配列シミュレーション", dist("bayer_pattern")),
            ("motion_blur_pixels", "モーションブラー量 (px)", rng_of("motion_blur_pixels")),
            ("chromatic_aberration_pixels", "色収差 (px)", rng_of("chromatic_aberration_pixels")),
            ("relative_noise_level", "相対ノイズ", rng_of("relative_noise_level")),
            ("aec_error", "自動露出誤差", rng_of("aec_error")),
            ("global_blur_std_dev", "全体ぼけ σ", rng_of("global_blur_std_dev")),
            ("flares", "レンズフレア", dist("flares")),
            ("upper_sign / lower_sign", "上下に併設された標識", dist("upper_sign", 5) + " / " + dist("lower_sign", 5)),
        ])
        body = f"""
<p class="lead">Synset Signset Germany（Fraunhofer IOSB / IPA, ITSC 2024）は、ドイツの標識 211 種類の 3D モデルを 327 枚の環境マップの中に置き、
OGRE（ラスタライズ）と Blender Cycles（パストレース）でレンダリングした合成データセットです（全体 105,500 枚、CC BY 4.0）。
GTSRB の 43 クラスに限定したサブセット（クラスあたり 500 枚、計 21,500 枚）が別リポジトリで公開され、合成データだけで学習した ConvNeXt が
実データの GTSRB テストで 98.3% に到達すると報告されています。ここでは Cycles 版 validation（{n:,} 枚、335x335 px）を使いました。
各サンプルは画像・標識マスク・セグメンテーション（上下の併設標識を含む）の 3 枚組で、レンダリング条件がメタデータとして付きます。
クラスごとの合成例は第 1 節の表の右端に、GTSRB の実例と並べて示しています。</p>
<h3>3 モダリティの実例（画像 / マスク / セグメンテーション、seed {self.seed}）</h3>
<div class="triples">{''.join(ex)}</div>
<table class="small full"><caption>表: validation {n:,} 枚のレンダリング条件の分布</caption><colgroup><col style="width:26%"><col style="width:20%"><col></colgroup><tr><th>列</th><th>意味</th><th>分布または範囲</th></tr>{meta_rows}</table>
<p class="note">合成データの用途: 実データの少ないクラスの補強、劣化条件（ブラー・ノイズ・露出）の体系的な評価、そして日本のように公開データが無い国の
標識カタログから学習データを生成する試みの雛形です。カードには EU AI 法の高リスク用途にそのまま使わないよう注意書きがあります。</p>
"""
        self.section("synset", "4. Synset Signset Germany: 合成データのモダリティとレンダリング条件", "ドイツ、合成、43 クラス版 21,500 枚（全体 211 クラス 105,500 枚）", body)

    # ---------------------------------------------------------------- TT100K
    def tt100k(self, n_frames: int = 4, n_crops: int = 3) -> None:
        root = self.data / "tt100k_hf"
        shards = sorted((root / "data").glob("*.parquet")) if root.exists() else []
        if not shards:
            print("skip tt100k: no data")
            return
        rng = self.rng("tt100k")
        cats = {}
        for line in (root / "categories.csv").read_text().splitlines()[1:]:
            i, _, name = line.split(",")
            cats[int(i)] = name
        # pass 1: objects only
        index: list[tuple[int, int, int, int]] = []  # (shard, row, width, height)
        boxes_by_cat: dict[int, list[tuple[int, int, list[float]]]] = collections.defaultdict(list)
        rows_with = []
        sides = []
        n_rows = 0
        for si, sh in enumerate(shards):
            pf = pq.ParquetFile(sh)
            for b in pf.iter_batches(batch_size=64, columns=["width", "height", "objects"]):
                d = b.to_pydict()
                for r in range(len(d["width"])):
                    gi = n_rows
                    n_rows += 1
                    objs = d["objects"][r]
                    index.append((si, r + 0, d["width"][r], d["height"][r]))
                    boxes = objs["bbox"] if isinstance(objs, dict) else [o["bbox"] for o in objs]
                    catsr = objs["category"] if isinstance(objs, dict) else [o["category"] for o in objs]
                    if boxes:
                        rows_with.append(gi)
                    for bb, c in zip(boxes, catsr):
                        boxes_by_cat[c].append((gi, c, bb))
                        sides.append(max(bb[2], bb[3]))
        # pick frames and crops
        frame_ids = rng.sample(rows_with, min(n_frames, len(rows_with)))
        crop_picks: dict[int, list[tuple[int, int, list[float]]]] = {}
        for c, lst in boxes_by_cat.items():
            crop_picks[c] = rng.sample(lst, min(n_crops, len(lst)))
        # pass 2: decode only the needed rows, keep crops (not frames) in memory
        crops_wanted: dict[int, list[tuple[int, list[float]]]] = collections.defaultdict(list)  # gi -> [(cat, bbox)]
        for c, lst in crop_picks.items():
            for gi_, _, bb in lst:
                crops_wanted[gi_].append((c, bb))
        frames_wanted = set(frame_ids)
        crops: dict[tuple[int, int, tuple], Image.Image] = {}
        frame_html = []
        gi = 0
        for si, sh in enumerate(shards):
            pf = pq.ParquetFile(sh)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["image"])
                col = tbl.column("image")
                for r in range(tbl.num_rows):
                    if gi in crops_wanted or gi in frames_wanted:
                        im = Image.open(io.BytesIO(col[r]["bytes"].as_py())).convert("RGB")
                        for c, bb in crops_wanted.get(gi, []):
                            crops[(gi, c, tuple(bb))] = crop_box(im, (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]), margin=0.15)
                        if gi in frames_wanted:
                            objs = [(c, bb) for c, lst in boxes_by_cat.items() for g, _, bb in lst if g == gi]
                            boxes = [((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]), "#e6194b", cats.get(c, str(c))) for c, bb in objs]
                            fr = draw_frame(im, boxes, 820)
                            cap = "、".join(f"{cats.get(c, c)}（{esc(TT100K_JA.get(cats.get(c, ''), ''))}, {bb[2]:.0f}x{bb[3]:.0f} px）" for c, bb in objs)
                            frame_html.append(f'<figure class="frame"><img src="{data_uri(fr, quality=78)}" alt=""><figcaption>{im.width}x{im.height}: {cap}</figcaption></figure>')
                            self.n_images += 1
                        del im
                    gi += 1
                del tbl, col
        # crop cards grouped by family
        def family(code: str) -> str:
            m = re.match(r"^(il|ip|iz|im|io|pl|pr|pm|ph|pw|pa|pne|pnl|pn|po|ps|pg|wo|i|p|w)", code)
            return m.group(1) if m else code
        order = ["p", "po", "pl", "pr", "pm", "ph", "pw", "pa", "pn", "pne", "pnl", "ps", "pg", "i", "io", "il", "ip", "iz", "im", "w", "wo"]
        cards_by_fam: dict[str, list[str]] = collections.defaultdict(list)
        present = 0
        for c in sorted(crop_picks, key=lambda c: cats.get(c, "")):
            code = cats.get(c, str(c))
            cards = []
            for gi_, _, bb in crop_picks[c]:
                cr = crops[(gi_, c, tuple(bb))]
                cards.append(f'<img src="{data_uri(fit(cr, 64))}" alt="">')
                self.n_images += 1
            present += 1
            gloss = TT100K_JA.get(code, "")
            cards_by_fam[family(code)].append(
                f'<div class="card wide"><div class="imgs">{"".join(cards)}</div><div class="cap"><b>{esc(code)}</b> {esc(gloss)}<br><span class="en">{len(boxes_by_cat[c])} 個</span></div></div>')
        fam_html = []
        for f in order + sorted(set(cards_by_fam) - set(order)):
            if cards_by_fam.get(f):
                fam_html.append(f'<h4>接頭辞 <code>{esc(f)}</code>（{len(cards_by_fam[f])} カテゴリ）</h4><div class="grid">{"".join(cards_by_fam[f])}</div>')
        fam_rows = "".join(f"<tr><td><code>{esc(a)}</code></td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in TT100K_FAMILIES)
        body = f"""
<p class="lead">Tsinghua-Tencent 100K（TT100K, CVPR 2016）は Tencent ストリートビューのパノラマから切り出した 2048x2048 px の画像 100,000 枚からなる
中国の標識検出・分類データセットで、約 10,000 枚に約 30,000 個の標識が矩形とピクセルマスク付きで注記されています（CC BY-NC、商用は著者に連絡）。
カテゴリ名は中国の国家標準 GB 5768 に沿ったコードで、接頭辞 p（禁令・禁止）、i（指示）、w（警告）に数字や制限値が続きます。
2016 版の注記ファイルには 221 カテゴリがあり、100 個以上の実例を持つ 45 カテゴリが論文のベンチマークに使われました。2021 版の追加注記で 232 カテゴリになっています。
ここで使った Hugging Face ミラー <code>finedet/tt100k</code>（232 カテゴリの categories.csv 付き）のうち {len(shards)} シャード、{n_rows:,} 枚、
矩形 {len(sides):,} 個（標識を含む画像 {len(rows_with):,} 枚）から抽出しました。カテゴリの日本語訳は 45 ベンチマーククラスを中心に作成者が付けた参考訳で、
数値つきコードはコード自体が意味です（pl60 は最高速度 60 km/h）。</p>
<table class="small"><caption>表: カテゴリコードの接頭辞と意味</caption><tr><th>接頭辞</th><th>意味</th><th>備考</th></tr>{fam_rows}</table>
<h3>フレームの実例（{len(frame_ids)} 枚、seed {self.seed}）</h3>
<div class="frames">{''.join(frame_html)}</div>
{hist_table(sides, f"表: 読み込んだ {len(sides):,} 個の矩形の長辺サイズ分布（2048 px 画像内）")}
<h3>カテゴリ別の切り抜き実例（本シャードに現れた {present} カテゴリ、各 {n_crops} 個まで、個数はシャード内）</h3>
{''.join(fam_html)}
"""
        self.section("tt100k", "5. TT100K: カテゴリコードの定義と実例", "中国、検出＋分類、100,000 枚、221/232 カテゴリ（45 が標準ベンチマーク）", body)
        self.sources.append(("TT100K (HF finedet/tt100k mirror, 3 shards)", "data/tt100k_hf", "CC BY-NC（公式）; ミラータグ cc-by-nc-4.0",
                             "https://cg.cs.tsinghua.edu.cn/traffic-sign/"))

    # ---------------------------------------------------------------- MTSD
    def mtsd(self, n_frames: int = 4, n_cards: int = 400) -> None:
        """MTSD from a sample of official images + annotation JSON (data/mtsd_official)."""
        root = self.data / "mtsd_official"
        ann_dir, img_dir = root / "annotations", root / "images"
        keys = sorted(p.stem for p in ann_dir.glob("*.json") if (img_dir / f"{p.stem}.jpg").exists()) if root.exists() else []
        if not keys:
            print("skip mtsd: no data")
            return
        rng = self.rng("mtsd")
        records = []  # (key, width, height, objects)
        boxes_by_label: dict[str, list[tuple[str, tuple[float, float, float, float], dict]]] = collections.defaultdict(list)
        sides = []
        props_count: collections.Counter = collections.Counter()
        n_pano = 0
        for k in keys:
            d = json.loads((ann_dir / f"{k}.json").read_text())
            objs = d.get("objects", [])
            records.append((k, d["width"], d["height"], objs))
            n_pano += int(bool(d.get("ispano")))
            for o in objs:
                b = o["bbox"]
                box = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                boxes_by_label[o["label"]].append((k, box, o.get("properties", {})))
                sides.append(max(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]))
                for pk, pv in o.get("properties", {}).items():
                    if pv:
                        props_count[pk] += 1
        n_boxes = len(sides)
        group_color = {"regulatory": "#d62728", "warning": "#ff7f0e", "information": "#1f77b4", "complementary": "#9467bd", "other-sign": "#7f7f7f"}

        def top(label: str) -> str:
            return label.split("--")[0]

        with_boxes = [r for r in records if r[3]]
        frames = rng.sample(with_boxes, min(n_frames, len(with_boxes)))
        frame_html = []
        for k, w, h, objs in frames:
            im = Image.open(img_dir / f"{k}.jpg").convert("RGB")
            boxes = [((o["bbox"]["xmin"], o["bbox"]["ymin"], o["bbox"]["xmax"], o["bbox"]["ymax"]), group_color.get(top(o["label"]), "#000"), "") for o in objs]
            fr = draw_frame(im, boxes, 820)
            names = collections.Counter(o["label"] for o in objs)
            cap = "、".join(f"{esc(lab)}{' x' + str(v) if v > 1 else ''}" for lab, v in names.most_common(8))
            frame_html.append(f'<figure class="frame"><img src="{data_uri(fr, quality=78)}" alt=""><figcaption>{w}x{h}, {len(objs)} 個: {cap}</figcaption></figure>')
            self.n_images += 1
            del im
        # one crop per label: prefer unoccluded boxes of at least 24 px
        picks: dict[str, tuple[str, tuple]] = {}
        for lab, lst in boxes_by_label.items():
            good = [x for x in lst if max(x[1][2] - x[1][0], x[1][3] - x[1][1]) >= 24 and not x[2].get("occluded")] or lst
            k, box, _ = rng.choice(good)
            picks[lab] = (k, box)
        by_key: dict[str, list[tuple[str, tuple]]] = collections.defaultdict(list)
        for lab, (k, box) in picks.items():
            by_key[k].append((lab, box))
        crops: dict[str, Image.Image] = {}
        for k, items in by_key.items():
            im = Image.open(img_dir / f"{k}.jpg").convert("RGB")
            for lab, box in items:
                crops[lab] = crop_box(im, box, margin=0.15)
            del im
        groups_html = []
        counts_by_group: collections.Counter = collections.Counter()
        for lab, lst in boxes_by_label.items():
            counts_by_group[top(lab)] += len(lst)
        for key, name, defi in MTSD_GROUPS:
            labs = sorted(((lab, len(lst)) for lab, lst in boxes_by_label.items() if top(lab) == key), key=lambda x: (-x[1], x[0]))
            cards = []
            for lab, cnt in labs[:n_cards]:
                short = lab.split("--", 1)[1] if "--" in lab else lab
                cards.append(f'<div class="card"><img src="{data_uri(fit(crops[lab], 64))}" alt=""><div class="cap">{esc(short)}<br><span class="en">{cnt} 個</span></div></div>')
                self.n_images += 1
            rest = ", ".join(esc(l.split("--", 1)[1] if "--" in l else l) for l, _ in labs[n_cards:])
            groups_html.append(
                f'<h3><span class="tag" style="background:{group_color[key]}">{key}</span> {esc(name)}: {esc(defi)}。標本中 {len(labs)} ラベル、{counts_by_group[key]:,} 個</h3>'
                f'<div class="grid">{"".join(cards)}</div>' + (f'<p class="rest">その他のラベル（各 1 例は省略）: {rest}</p>' if rest else ""))
        prop_rows = "".join(f"<tr><td><code>{esc(k)}</code></td><td>{esc(MTSD_PROPS.get(k, ''))}</td><td class='num'>{v:,}</td><td class='num'>{100 * v / max(1, n_boxes):.1f}%</td></tr>"
                            for k, v in sorted(props_count.items(), key=lambda x: -x[1]))
        res = collections.Counter(f"{w}x{h}" for _, w, h, _ in records)
        body = f"""
<p class="lead">Mapillary Traffic Sign Dataset（MTSD, ECCV 2020）は世界中の Mapillary 投稿画像約 10 万枚に標識の矩形を付けたもので、
完全注記 52,453 枚・257,543 個、313 クラス（v2 では 400 クラス）＋ タクソノミー外の other-sign からなります（CC BY-NC-SA 4.0、テスト正解は非公開）。
ラベルは <code>大分類--意味--gN</code> の形式で、gN は国ごとの見た目の違い（グラフィックのバリエーション）を区別する番号です。
各矩形には occluded（遮蔽）、out-of-frame、ambiguous、exterior、included、direction-or-information、dummy などの属性が付きます。
ここでは公式の注記 JSON と画像を Hugging Face ミラー（NgTMDuc/cl_dataset 内の mtsd_v2_fully_annotated）から seed 固定で {len(records):,} 枚標本抽出しました
（矩形 {n_boxes:,} 個、パノラマ {n_pano} 枚、解像度は {", ".join(f"{k} {v} 枚" for k, v in res.most_common(3))} など）。
大分類ごとに、標本に現れたラベルを個数順に 1 例ずつ示します（遮蔽なし・24 px 以上の矩形を優先）。</p>
<h3>フレームの実例（{len(frames)} 枚、seed {self.seed}、枠の色は大分類）</h3>
<div class="frames">{''.join(frame_html)}</div>
<div class="side">{hist_table(sides, f"表: 標本 {n_boxes:,} 個の矩形の長辺サイズ分布")}
<table class="small"><caption>表: 矩形属性（True の個数）</caption><tr><th>属性</th><th>意味</th><th>個数</th><th>割合</th></tr>{prop_rows}</table></div>
{''.join(groups_html)}
"""
        self.section("mtsd", "6. Mapillary Traffic Sign Dataset (MTSD): タクソノミーと実例", "世界、検出＋分類、約 10 万枚、313/400 クラス", body)
        self.sources.append(("MTSD（公式注記 JSON と画像の標本; HF NgTMDuc/cl_dataset ミラー）", "data/mtsd_official",
                             "CC BY-NC-SA（同梱 LICENSE.txt）", "https://www.mapillary.com/dataset/trafficsign"))

    # ---------------------------------------------------------------- CURE-TSR
    def cure(self, n_per_sign: int = 5, levels: tuple[int, ...] = (1, 3, 5)) -> None:
        zp = self.data / "cure_tsr" / "CURE-TSR.zip"
        if not zp.exists():
            print("skip cure: no data")
            return
        try:
            outer = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print("skip cure: zip incomplete")
            return
        rng = self.rng("cure")
        # The Zenodo archive holds four stored (uncompressed) zips: Real_Train, Real_Test, Unreal_Train, Unreal_Test.
        inner: dict[str, zipfile.ZipFile] = {}
        for member in outer.namelist():
            if member.lower().endswith(".zip"):
                inner[member] = zipfile.ZipFile(outer.open(member))
        pat = re.compile(r"(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d+)\.bmp$")
        parsed = []  # (inner zip, name, seq, sign, challenge, level)
        split_counts: collections.Counter = collections.Counter()
        for member, zf in inner.items():
            for n in zf.namelist():
                m = pat.search(n)
                if m:
                    parsed.append((member, n, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
                    split_counts[member.replace(".zip", "")] += 1
        by_key: dict[tuple[int, int, int, int], list[tuple[str, str]]] = collections.defaultdict(list)
        for member, n, seq, sign, ch, lvl in parsed:
            by_key[(seq, sign, ch, lvl)].append((member, n))
        counts = collections.Counter((seq, sign) for _, _, seq, sign, ch, lvl in parsed if ch == 0)

        def img(item: tuple[str, str]) -> Image.Image:
            member, n = item
            return Image.open(io.BytesIO(inner[member].read(n))).convert("RGB")

        rows = []
        for s in range(1, 15):
            cells = []
            for seq in (1, 2):
                pool = by_key.get((seq, s, 0, 0), [])
                pick = rng.sample(pool, min(n_per_sign, len(pool)))
                cells.append("".join(thumb_html(img(n), 56, n[1]) for n in pick))
                self.n_images += len(pick)
            rows.append(f"<tr><td class='num'>{s:02d}</td><td><b>{esc(CURE_SIGNS_JA[s - 1])}</b><br><span class='en'>{esc(CURE_SIGNS[s - 1])}</span></td>"
                        f"<td class='num'>{counts[(1, s)]:,}<br>{counts[(2, s)]:,}</td><td class='thumbs'>{cells[0]}</td><td class='thumbs'>{cells[1]}</td></tr>")
        # challenge matrix for two sign types
        matrix = []
        for seq, label in ((1, "実 (real)"), (2, "合成 (unreal)")):
            sign = 1 if seq == 1 else 6
            head = "".join(f"<th>Lv {l}</th>" for l in levels)
            mrows = []
            # the same source image (same index) across every challenge and level, when the file exists
            base_pool = by_key.get((seq, sign, 0, 0), [])
            # prefer a large crop (top 10 percent by file size) so that the degradations are visible
            base_pool = sorted(base_pool, key=lambda x: -inner[x[0]].getinfo(x[1]).file_size)[: max(1, len(base_pool) // 10)]
            base = rng.choice(base_pool) if base_pool else None
            base_idx = pat.search(base[1]).group(5) if base else None
            if base:
                cells0 = f"<td>{thumb_html(img(base), 56, base[1])}</td>" + "".join("<td></td>" for _ in levels[1:])
                self.n_images += 1
                mrows.append(f"<tr><td class='num'>00</td><td>{esc(CURE_CHALLENGES_JA[0])}<br><span class='en'>{esc(CURE_CHALLENGES[0])}</span></td>{cells0}</tr>")
            for ch in range(1, 13):
                cells = []
                for l in levels:
                    pool = by_key.get((seq, sign, ch, l), [])
                    same = [x for x in pool if base_idx and x[1].endswith(f"_{base_idx}.bmp")]
                    if same or pool:
                        n = same[0] if same else rng.choice(pool)
                        cells.append(f"<td>{thumb_html(img(n), 56, n[1])}</td>")
                        self.n_images += 1
                    else:
                        cells.append("<td></td>")
                mrows.append(f"<tr><td class='num'>{ch:02d}</td><td>{esc(CURE_CHALLENGES_JA[ch])}<br><span class='en'>{esc(CURE_CHALLENGES[ch])}</span></td>{''.join(cells)}</tr>")
            matrix.append(f"<table class='cls half'><caption>表: {label} シーケンス、標識 {sign:02d} {esc(CURE_SIGNS_JA[sign - 1])} の劣化種別 x レベル</caption>"
                          f"<thead><tr><th>#</th><th>劣化</th>{head}</tr></thead><tbody>{''.join(mrows)}</tbody></table>")
        n_total = len(parsed)
        body = f"""
<p class="lead">CURE-TSR（Georgia Tech OLIVES, NeurIPS 2017 workshop）は、劣化条件下の標識分類の頑健性を測るデータセットです。
実シーケンス（BelgiumTS の映像から切り出し）と Unreal Engine 4 で合成したシーケンスの標識画像に、12 種類の劣化を 5 段階の強さで加えています
（Adobe After Effects で後処理）。クラスは 14 種類の粗い標識タイプで、速度制限はすべて 1 クラスにまとめられています。
ファイル名は <code>シーケンス種別_標識種別_劣化種別_劣化レベル_連番.bmp</code>（01 実 / 02 合成、劣化 00 はなし）。
Zenodo の配布物（4.9 GB、CC BY 4.0、内訳 {", ".join(f"{k} {v:,}" for k, v in sorted(split_counts.items()))}）から {n_total:,} 枚のファイル名を解析し、
劣化なし画像の実例と、劣化の種類とレベルの行列を示します。切り抜きは 3x7 から 206x277 px と小さく、多くが 30 px 未満です。</p>
<table class="cls"><caption>表: 14 の標識タイプ。枚数は劣化なし（実 / 合成）、実例は劣化なし画像から seed {self.seed} で {n_per_sign} 枚</caption>
<thead><tr><th>#</th><th>標識タイプ</th><th>実<br>合成</th><th>実例（実）</th><th>実例（合成）</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="side">{''.join(matrix)}</div>
<p class="note">行列は同じ元画像（同じ連番）に 12 種類の劣化を 3 段階で加えたもので、該当ファイルが無い場合だけ別の画像で代用しています。
読み方: 論文ではコーデック誤り（03）がレベル 1 でも 30% 以上精度を落とす最も有害な劣化で、合成シーケンスの劣化画像を学習に加えると
84 組中 72 組で精度が改善したと報告されています。実画像は BelgiumTS 由来なので、BTSC と CURE-TSR は独立なデータではありません。</p>
"""
        self.section("cure", "7. CURE-TSR: 14 標識タイプと 12 劣化 x 5 レベル（実と合成）", "ベルギー実映像＋合成、分類（頑健性）、約 220 万枚、14 クラス", body)
        self.sources.append(("CURE-TSR (Zenodo record 3903066)", "data/cure_tsr/CURE-TSR.zip", "CC BY 4.0（Zenodo）; GitHub 版は利用条件フォーム",
                             "https://zenodo.org/records/3903066"))

    # ---------------------------------------------------------------- Japan
    def japan(self, json_path: Path | None, overview: Path | None = None) -> None:
        if not json_path or not json_path.exists():
            print("skip japan: no json")
            return
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        have = [e for e in entries if e.get("png") and Path(e["png"]).exists() and Path(e["png"]).stat().st_size > 100]
        by_sec: dict[str, list[dict]] = collections.defaultdict(list)
        for e in have:
            by_sec[e["section"]].append(e)
        parts = []
        def num_key(e: dict) -> tuple:
            n = e.get("number") or "999"
            m = re.match(r"(\d{3})(.*)", n)
            return (int(m.group(1)), m.group(2)) if m else (999, n)

        for sec, defi in JP_CATEGORY_DEF:
            items = sorted(by_sec.get(sec, []), key=num_key)
            cards = []
            for e in items:
                im = Image.open(e["png"]).convert("RGB")
                name = e.get("name_wiki") or e.get("name") or ""
                cards.append(f'<div class="card jp"><img src="{data_uri(fit(im, 110, upscale=False), quality=88)}" alt="">'
                             f'<div class="cap"><b>{esc(e.get("number") or "")}</b> {esc(name)}</div></div>')
                self.n_images += 1
            parts.append(f"<h3>{esc(sec)}（{len(items)} 図柄）: {esc(defi)}</h3><div class='grid'>{''.join(cards)}</div>")
        ov_html = ""
        if overview and overview.exists():
            ov = Image.open(overview).convert("RGB")
            ov = fit(ov, 1600, upscale=False)
            ov_html = (f'<figure class="overview"><img src="{data_uri(ov, quality=80)}" alt="">'
                       f'<figcaption>国土交通省 道路局「道路標識一覧」（A3 1 枚、ichiran.pdf）の全体。以下の図柄はこの一覧からラベル位置をもとに切り出したものです。</figcaption></figure>')
            self.n_images += 1
        body = f"""
<p class="lead">日本の道路標識は「道路標識、区画線及び道路標示に関する命令」（道路標識令、昭和 35 年総理府・建設省令第 3 号）の別表で定義され、
本標識 4 種（案内・警戒・規制・指示）と補助標識に分かれます（e-Gov 法令検索の別表を解析すると 163 種類、様式違いを含めて 237 図柄）。
日本の標識については、研究用途で公開されたデータセットが存在しません（技術報告 第 3 節: Roboflow の数十枚規模の案件と、学習データを非公開にした
infra-intelligence の 5 分類 YOLOX-tiny 検出器のみ）。そのため本節は実写データではなく、国土交通省 道路局が公開している「道路標識一覧」の公式図柄
{len(have)} 点（法令の別表に由来する図柄のため著作権法第 13 条により著作権の対象外）を分類ごとに示し、OpenTSR が日本向けに収集・生成すべきクラスの一覧として使います。
番号と名称は一覧の表記に従い、日本語版 Wikipedia「日本の道路標識」の整理で補いました（補助標識は代表的な記載例、案内標識は様式違いを含みます）。</p>
{ov_html}
<table class="small"><caption>表: 道路標識令の 5 分類（infra-intelligence の公開検出器もこの 5 分類をクラスとしています）</caption>
<tr><th>分類</th><th>定義</th></tr>{''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>' for a, b in JP_CATEGORY_DEF)}</table>
{''.join(parts)}
<p class="note">OpenTSR での扱い: 日本の細分類（規制 1 種類ごと）を学習するには、自前撮影に加え、これらの図柄を Synset Signset と同じ手順で
3D レンダリングする合成データ、Panoramax 型の市民投稿画像、VLM による自動ラベリングを組み合わせる計画です（技術報告 第 7 節）。</p>
"""
        self.section("japan", f"8. 日本の道路標識（道路標識令）: 分類と公式図柄（{len(have)} 図柄）",
                     "日本、公開データセットなし、標識令 163 種類 / 237 図柄", body)
        self.sources.append(("日本の道路標識 公式図柄（国土交通省 道路局「道路標識一覧」）", "catalog_src/mlit/ichiran.pdf（300 dpi で切り出し）",
                             "法令の別表に由来する図柄（著作権法第 13 条）; 一覧 PDF は国土交通省の公開資料",
                             "https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf"))

    # ---------------------------------------------------------------- assemble
    def render(self, out_html: Path, title: str) -> None:
        today = dt.date.today().isoformat()
        toc = "".join(f'<tr><td><a href="#{a}">{esc(t)}</a></td><td>{esc(s)}</td></tr>' for a, t, s in self.toc)
        src = "".join(f"<tr><td>{esc(a)}</td><td><code>{esc(b)}</code></td><td>{esc(c)}</td><td class='en'>{esc(d)}</td></tr>" for a, b, c, d in self.sources)
        front = f"""
<header>
<h1>{esc(title)}</h1>
<p class="meta">OpenTSR プロジェクト（pingfoo/OpenTSR）· 生成日 {today} · 乱数 seed {self.seed} · 埋め込み画像 {self.n_images:,} 枚<br>
生成スクリプト: <code>scripts/make_dataset_catalog.py</code>（Claude Mythos 5.1 と共同作成）</p>
</header>
<section class="intro">
<h2>はじめに: このカタログの読み方</h2>
<p>本カタログは、道路標識認識の調査（技術報告 2026-10-31）で「代表的」と判断した公開データセットについて、<b>クラスやカテゴリの定義</b>と
<b>実際にサンプリングした画像</b>を並べて示すものです。数字だけの表では分からない「同じ『速度制限』でも国や年代で見た目がどう違うか」
「標識が画面上で何ピクセルか」「ラベル体系の粒度がどれほど違うか」を、モデル設計と評価設計の前に目で確認することが目的です。</p>
<ul>
<li><b>サンプリング</b>: 各データセットのローカルコピー（技術報告の再現実験と同じ Hugging Face / Zenodo のミラー）から、Python の
<code>random.Random("seed:dataset")</code> による固定乱数で抽出しています。選び方に人手は入っていないので、劣化した画像や小さな画像もそのまま載っています。</li>
<li><b>切り抜き</b>: 検出データセットの実例は正解矩形の周囲に 15% の余白を付けて切り出し、長辺 64 から 72 px に縮小（小さいものは拡大）しています。
フレームは幅 820 px に縮小し、枠の色はカテゴリを表します。</li>
<li><b>定義</b>: 公式のクラス名・コードはそのまま載せ、日本語訳は作成者による参考訳です。ドイツ StVO 番号や中国 GB 5768 の対応は目安です。</li>
<li><b>ライセンス</b>: 画像は各データセットのライセンスに従います（巻末の表）。TT100K と MTSD は非商用ライセンスであり、本カタログでの掲載は研究目的の引用です。
OpenTSR が再配布するデータセットには含めません。</li>
</ul>
<table class="small full"><caption>表: 代表的データセットの概要（詳細と出典は docs/datasets.md と技術報告 第 3 節）</caption>
<tr><th>データセット</th><th>国・地域</th><th>用途</th><th>規模</th><th>クラス数</th><th>ライセンス</th><th>本カタログで使用</th></tr>
{''.join(f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td><td>{esc(e)}</td><td>{esc(f)}</td><td>{esc(g)}</td></tr>" for a, b, c, d, e, f, g in OVERVIEW)}</table>
<table class="small full"><caption>表: 本カタログの構成</caption><tr><th>節</th><th>概要</th></tr>{toc}</table>
</section>
"""
        appendix = f"""
<section class="ds" id="appendix"><h2>付録: 出典、ライセンス、再現方法</h2>
<table class="small"><caption>表: 本カタログで使用したデータとライセンス</caption>
<tr><th>データセット</th><th>ローカルパス（リポジトリ内、gitignore 対象）</th><th>ライセンス（調査時点の記載）</th><th>出典 URL</th></tr>{src}</table>
<h3>本カタログに収録しなかった代表的データセット</h3>
<table class="small full"><caption>表: 調査（docs/datasets.md）で挙げたが本版ではサンプリングしなかったもの</caption>
<tr><th>データセット</th><th>国</th><th>クラス数 / 規模</th><th>ライセンス</th><th>未収録の理由</th></tr>
{''.join(f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td><td>{esc(e)}</td></tr>" for a, b, c, d, e in NOT_INCLUDED)}</table>
<h3>再現方法</h3>
<pre>make gtsrb gtsdb                       # GTSRB と GTSDB のミラー取得（検証付き）
hf download Panoramax/classified_de_road_signs --repo-type dataset --local-dir data/panoramax_de   # validation のみでも可
hf download FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset cycles/validation.parquet --repo-type dataset --local-dir data/synset_signset
hf download finedet/tt100k data/test-00003-of-00004.parquet data/train-00005-of-00007.parquet data/train-00006-of-00007.parquet categories.csv --repo-type dataset --local-dir data/tt100k_hf
python scripts/sample_mtsd_official.py --n 320      # official MTSD JSON + images from the NgTMDuc/cl_dataset mirror into data/mtsd_official
curl -L -o data/cure_tsr/CURE-TSR.zip https://zenodo.org/records/3903066/files/CURE-TSR.zip
python scripts/make_dataset_catalog.py --out runs/catalog --seed {self.seed} --pdf</pre>
<p class="note">同じ seed とデータファイルを使えば同じ画像が選ばれます。ミラーのシャード構成が変わると TT100K と MTSD の抽出結果は変わります。</p>
</section>
"""
        css = CSS
        doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>{esc(title)}</title><style>{css}</style></head>
<body>{front}{''.join(self.parts)}{appendix}</body></html>"""
        out_html.write_text(doc, encoding="utf-8")


CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: "Noto Sans CJK JP", "Noto Sans JP", sans-serif; font-size: 9.2pt; line-height: 1.5; color: #111; }
header h1 { font-size: 19pt; margin: 0 0 6px 0; line-height: 1.35; }
.meta { font-size: 8.6pt; color: #444; margin: 0 0 14px 0; }
h2 { font-size: 13.5pt; margin: 0 0 8px 0; padding-bottom: 3px; border-bottom: 2px solid #333; }
section.ds { page-break-before: always; }
section.intro { margin-bottom: 10px; }
h3 { font-size: 10.4pt; margin: 12px 0 4px 0; page-break-after: avoid; }
h4 { font-size: 9.4pt; margin: 8px 0 3px 0; page-break-after: avoid; color: #333; }
p { margin: 0 0 6px 0; text-align: justify; }
p.lead { font-size: 9.2pt; }
p.note { font-size: 8.4pt; color: #333; background: #f5f5f5; padding: 5px 8px; border-left: 3px solid #999; }
p.rest { font-size: 7.6pt; color: #444; line-height: 1.4; }
ul { margin: 0 0 6px 0; padding-left: 1.4em; }
li { margin-bottom: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.35; margin: 6px 0 8px 0; }
table caption { caption-side: top; text-align: left; font-size: 8.4pt; font-weight: 700; margin-bottom: 3px; }
th, td { border: 1px solid #999; padding: 2px 4px; vertical-align: middle; text-align: left; }
th { background: #eee; font-weight: 700; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.thumbs { white-space: nowrap; padding: 2px 3px; }
img.th { height: 44px; width: 44px; object-fit: contain; margin: 1px; vertical-align: middle; background: #fafafa; border: 1px solid #ddd; }
img.th.syn { border-color: #2a9d8f; }
table.cls td { font-size: 7.8pt; }
table.small { width: auto; min-width: 60%; page-break-inside: avoid; }
table.small.full { width: 100%; }
th.nw { white-space: nowrap; }
p.capt { font-size: 8.4pt; font-weight: 700; margin: 6px 0 2px 0; }
table.small.hist td { padding: 1px 5px; }
table.half { width: 48%; display: inline-table; vertical-align: top; margin-right: 1%; }
.bar { height: 8px; background: #4a7ab5; }
.tag { display: inline-block; color: #fff; font-size: 7.4pt; padding: 0 4px; border-radius: 3px; font-weight: 700; }
.en { color: #555; font-size: 7.4pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7.8pt; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt; background: #f5f5f5; padding: 6px 8px; white-space: pre-wrap; word-break: break-all; }
.frames { display: flex; flex-wrap: wrap; gap: 8px; }
figure.frame { margin: 0 0 6px 0; width: 49%; page-break-inside: avoid; }
figure.frame img { width: 100%; border: 1px solid #ccc; }
figcaption { font-size: 7.4pt; color: #333; line-height: 1.3; margin-top: 2px; }
.grid { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.card { width: 84px; border: 1px solid #ddd; padding: 3px; page-break-inside: avoid; background: #fff; }
.card img { height: 56px; width: 100%; object-fit: contain; display: block; margin: 0 auto 2px auto; background: #fafafa; }
.card .cap { font-size: 6.8pt; line-height: 1.25; text-align: center; word-break: break-all; }
.card.wide { width: 118px; }
.card.wide .imgs { display: flex; justify-content: center; gap: 1px; }
.card.wide .imgs img { width: 36px; height: 36px; object-fit: contain; }
.card.jp { width: 96px; }
.card.jp img { height: 60px; }
figure.overview { margin: 4px 0 8px 0; width: 100%; }
figure.overview img { width: 100%; border: 1px solid #ccc; }
.card.jp .cap { font-size: 6.6pt; }
.triples { display: flex; flex-wrap: wrap; gap: 8px; }
.triple { width: 48%; border: 1px solid #ddd; padding: 4px; page-break-inside: avoid; }
.triple img { width: 31%; margin-right: 1.5%; height: 100px; object-fit: contain; background: #fafafa; }
.triple .cap { font-size: 7.2pt; color: #333; line-height: 1.3; }
ul.labels { columns: 3; font-size: 6.6pt; line-height: 1.3; list-style: none; padding: 0; font-family: "DejaVu Sans Mono", monospace; }
ul.labels li { break-inside: avoid; }
.side { display: flex; gap: 2%; align-items: flex-start; }
.side table { width: 49%; min-width: 0; }
a { color: #0b4f8a; text-decoration: none; }
"""


def render_pdf(html_path: Path, pdf_path: Path, footer: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + str(html_path.resolve()))
        pg.wait_for_timeout(800)
        pg.pdf(path=str(pdf_path), format="A4", print_background=True, display_header_footer=True,
               header_template="<div></div>",
               footer_template=f'<div style="font-size:7.5px; width:100%; text-align:center; font-family: sans-serif; color:#444;">{html.escape(footer)} · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
               margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"})
        b.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/catalog")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma separated subset of: gtsrb,gtsdb,panoramax,synset,tt100k,mtsd,cure,japan")
    ap.add_argument("--japan-json", default="", help="JSON with the Japanese sign tiles (section, number, name, png)")
    ap.add_argument("--japan-overview", default="", help="PNG of the whole MLIT sign list page (optional)")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--title", default="OpenTSR 道路標識データセット・カタログ: 分類の定義と実例")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = Catalog(args.seed, Path(args.data))
    only = set(filter(None, args.only.split(",")))

    def want(name: str) -> bool:
        return not only or name in only

    if want("gtsrb"):
        cat.gtsrb()
    if want("gtsdb"):
        cat.gtsdb()
    if want("panoramax"):
        cat.panoramax()
    if want("synset"):
        cat.synset()
    if want("tt100k"):
        cat.tt100k()
    if want("mtsd"):
        cat.mtsd()
    if want("cure"):
        cat.cure()
    if want("japan"):
        cat.japan(Path(args.japan_json) if args.japan_json else None, Path(args.japan_overview) if args.japan_overview else None)

    html_path = out / "catalog.html"
    cat.render(html_path, args.title)
    print(f"wrote {html_path} ({html_path.stat().st_size / 1e6:.1f} MB, {cat.n_images} images)")
    if args.pdf:
        pdf_path = out / "catalog.pdf"
        render_pdf(html_path, pdf_path, f"OpenTSR 道路標識データセット・カタログ {dt.date.today().isoformat()}")
        print(f"wrote {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
