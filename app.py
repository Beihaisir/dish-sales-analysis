# app.py (V7.3 - 指定“一天内时段窗口”占比饼图 + 增强分类规则)
# 你要的逻辑：在你【查询】选定的日期范围内，再选一个“一天内时段窗口”（如 11:30-14:30），
# 计算：窗口内销量 / 查询范围总销量，并用饼图展示（窗口内 vs 窗口外）
import re
import io
import json
import hashlib
import datetime as dt
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# 基础配置
# =========================
st.set_page_config(page_title="菜品销售分析（正大餐饮）", layout="wide")
st.title("菜品销售分析")

# =========================
# 默认分类映射（已合并新增未命中菜品的分类建议）
# =========================
DEFAULT_CATEGORY_MAP = {
    "板筋类": ["板筋"],
    "猪肝类": ["猪肝"],
    "鸡丁类": ["鸡丁"],
    "卤蛋类": ["卤蛋", "卤鸡蛋"],
    "煎蛋类": ["煎蛋"],

    # ✅ 饮料类（扩充）
    "饮料类": [
        "饮料", "可乐", "雪碧", "美年达", "农夫山泉",
        "北冰洋", "红牛", "加多宝", "唯怡豆奶", "椰汁", "苹果醋",
        "橙汁", "芒果汁", "杨梅", "小青柠", "小蒙牛",
        "矿泉水", "正大矿泉水", "北六柠百美",
        "花生味6点半", "玻璃瓶", "听装", "生榨", "饮品", "饭后半小时"
    ],

    "牛肉类": ["牛肉"],
    "腰花类": ["腰花"],
    "鸡胗类": ["鸡胗"],
    "肥肠类": ["肥肠"],
    "鸡杂类": ["鸡杂"],
    "双脆类": ["双脆"],
    "即食类": ["三鲜", "老母鸡"],

    # ✅ 新增分类
    "卤味类": ["卤豆腐", "卤鸡腿", "卤猪脚", "卤鸡脚", "小卤拼", "特色小卤拼"],
    "鳝鱼类": ["鳝鱼"],
    "掌中宝类": ["掌中宝"],
    "老三丁类": ["老三丁"],
    "五花肉类": ["五花肉"],
    "主食点心类": ["红糖馒头", "鲜肉包", "流沙包", "小笼包", "蒸饺", "烧麦"],
    "鱼香肉丝类": ["鱼香肉丝"],

    "其他类": ["打包盒", "单份米饭", "加面", "鸡排", "凑价"],
}

# 蛋类固定单价（营业额修正）
FIXED_PRICE_CATEGORY = {"卤蛋类": 2.0, "煎蛋类": 2.0}
EGG_CATEGORIES = set(FIXED_PRICE_CATEGORY.keys())

# 做法单加白名单（出现一次记一次 × 菜品数量）
ADDON_WHITELIST = [
    "加鸡丁", "加牛肉", "加板筋", "加腰花",
    "加猪肝", "加鸡胗", "加肥肠", "加鸡杂",
    "打包"
]


# =========================
# 工具函数
# =========================
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_topn_with_others(df: pd.DataFrame, name_col: str, value_col: str, topn: int = 20) -> pd.DataFrame:
    df = df.sort_values(value_col, ascending=False).copy()
    if len(df) <= topn:
        return df
    top = df.head(topn).copy()
    others_sum = df.iloc[topn:][value_col].sum()
    others = pd.DataFrame({name_col: ["其他"], value_col: [others_sum]})
    return pd.concat([top, others], ignore_index=True)

def parse_rule_table(rule_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    规则表要求至少两列：分类, 关键词（中文列名需严格一致）
    支持一个分类对应多关键词（多行）
    """
    cols = [c.strip() for c in rule_df.columns.astype(str)]
    rule_df.columns = cols

    if "分类" not in rule_df.columns or "关键词" not in rule_df.columns:
        raise ValueError("规则表必须包含列：分类、关键词（严格中文列名）")

    rule_df = rule_df[["分类", "关键词"]].copy()
    rule_df["分类"] = rule_df["分类"].map(normalize_str)
    rule_df["关键词"] = rule_df["关键词"].map(normalize_str)
    rule_df = rule_df[(rule_df["分类"] != "") & (rule_df["关键词"] != "")]

    out: Dict[str, List[str]] = {}
    for cat, g in rule_df.groupby("分类"):
        kws = sorted(set(g["关键词"].tolist()))
        out[cat] = kws

    if not out:
        raise ValueError("规则表解析后为空，请检查内容。")
    return out

def build_time_bucket(df: pd.DataFrame, minutes: int) -> pd.Series:
    t = pd.to_datetime(df["创建时间"], errors="coerce")
    return t.dt.floor(f"{minutes}T")

@st.cache_data(show_spinner=False)
def read_excel_safely(file_bytes: bytes) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    preview = pd.read_excel(bio, sheet_name=0, header=None, nrows=40)

    header_row = None
    for i in range(len(preview)):
        row = preview.iloc[i].astype(str).tolist()
        if ("创建时间" in row) and ("菜品名称" in row) and ("菜品数量" in row):
            header_row = i
            break
    if header_row is None:
        raise ValueError("未找到表头行（需要包含：创建时间/菜品名称/菜品数量）。")

    bio.seek(0)
    df = pd.read_excel(bio, sheet_name=0, header=header_row)

    needed = ["创建时间", "菜品名称", "菜品数量", "规格名称", "做法", "优惠后小计价格"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{missing}")

    return df[needed].copy()

@st.cache_data(show_spinner=False)
def read_csv_safely_generic(file_bytes: bytes, encoding_guess: str = "utf-8") -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio, encoding=encoding_guess)
    except Exception:
        bio.seek(0)
        df = pd.read_csv(bio, encoding="gbk")
    return df

@st.cache_data(show_spinner=False)
def read_data_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = read_csv_safely_generic(file_bytes)
    else:
        df = read_excel_safely(file_bytes)

    needed = ["创建时间", "菜品名称", "菜品数量", "规格名称", "做法", "优惠后小计价格"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{missing}")
    return df[needed].copy()

def compress_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["创建时间"] = pd.to_datetime(df["创建时间"], errors="coerce")
    df["菜品数量"] = pd.to_numeric(df["菜品数量"], errors="coerce").fillna(0).astype("int32")
    df["优惠后小计价格"] = pd.to_numeric(df["优惠后小计价格"], errors="coerce").fillna(0).astype("float64")
    for c in ["规格名称", "菜品名称", "做法"]:
        df[c] = df[c].astype(str).fillna("")
        df[c] = df[c].astype("string")
    return df

def build_category_long_df_and_coverage(df: pd.DataFrame, category_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    多分类展开（双计数）
    - 分类基础：按关键词匹配得到的分类
    - 分类（展示分类）：
        * 菜品名称以“加”开头 => 单加-<菜品名称>（更直观，不混入大类）
        * 否则 => 分类基础
    """
    names = df["菜品名称"].astype(str)

    idx_list, cat_list = [], []
    any_hit_mask = np.zeros(len(df), dtype=bool)

    for cat, keywords in category_map.items():
        if not keywords:
            continue
        pattern = "(" + "|".join(re.escape(k) for k in keywords) + ")"
        mask = names.str.contains(pattern, regex=True, na=False).to_numpy()

        any_hit_mask |= mask
        hit_idx = np.flatnonzero(mask)
        if hit_idx.size == 0:
            continue
        idx_list.append(hit_idx)
        cat_list.append(np.repeat(cat, hit_idx.size))

    if idx_list:
        all_idx = np.concatenate(idx_list)
        all_cat = np.concatenate(cat_list)
        long_df = df.iloc[all_idx].copy()
        long_df["分类基础"] = all_cat
    else:
        long_df = df.iloc[0:0].assign(分类基础=pd.Series(dtype="object"))

    unmatched = df.loc[~any_hit_mask, ["创建时间", "菜品名称", "菜品数量", "优惠后小计价格", "规格名称", "做法"]].copy()

    coverage = pd.DataFrame({
        "总行数": [len(df)],
        "命中行数": [int(any_hit_mask.sum())],
        "未命中行数": [int((~any_hit_mask).sum())],
        "命中率": [float(any_hit_mask.mean()) if len(df) else 0.0]
    })

    dish_name = long_df["菜品名称"].astype(str).str.strip()
    long_df["是否加菜品"] = dish_name.str.startswith("加")
    long_df["分类"] = np.where(long_df["是否加菜品"], "单加-" + dish_name, long_df["分类基础"])

    return long_df, unmatched, coverage

def compute_addon_summary_vectorized(cat_df: pd.DataFrame, addon_list: list) -> pd.DataFrame:
    """做法单加：出现次数 × 菜品数量；按【展示分类】汇总"""
    if cat_df.empty:
        return pd.DataFrame(columns=["分类", "单加项", "数量"])

    method = cat_df["做法"].astype(str).fillna("")
    qty = cat_df["菜品数量"].astype("int64")

    parts = []
    for addon in addon_list:
        counts = method.str.count(re.escape(addon)).astype("int64")
        contrib = counts * qty
        s = contrib.groupby(cat_df["分类"]).sum()
        tmp = s.rename("数量").reset_index()
        tmp["单加项"] = addon
        tmp = tmp[["分类", "单加项", "数量"]]
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)
    out = out[out["数量"] > 0].sort_values("数量", ascending=False)
    return out

def rule_health_check(df_q: pd.DataFrame, category_map: dict, unmatched_df: pd.DataFrame, cat_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unmatched_top = (
        unmatched_df.groupby("菜品名称", as_index=False)
        .agg(未命中行数=("菜品名称", "size"),
             数量合计=("菜品数量", "sum"),
             小计合计=("优惠后小计价格", "sum"))
        .sort_values(["数量合计", "小计合计"], ascending=False)
        .head(50)
    )

    multi_hit = (
        cat_df.groupby(["创建时间", "菜品名称", "菜品数量", "规格名称", "做法", "优惠后小计价格", "时间段"], as_index=False)["分类基础"]
        .nunique()
        .rename(columns={"分类基础": "命中分类数"})
    )
    multi_hit_top = multi_hit.sort_values(["命中分类数", "菜品数量"], ascending=[False, False]).head(50)

    names = df_q["菜品名称"].astype(str)
    rows = []
    for cat, kws in category_map.items():
        for kw in kws:
            mask = names.str.contains(re.escape(kw), na=False)
            hit_dishes = df_q.loc[mask, "菜品名称"].astype(str).nunique()
            hit_rows = int(mask.sum())
            if hit_rows > 0:
                rows.append({"分类": cat, "关键词": kw, "命中菜品数": int(hit_dishes), "命中行数": hit_rows})
    kw_wide = pd.DataFrame(rows)
    if not kw_wide.empty:
        kw_wide = kw_wide.sort_values(["命中菜品数", "命中行数"], ascending=False).head(100)
    else:
        kw_wide = pd.DataFrame(columns=["分类", "关键词", "命中菜品数", "命中行数"])

    return unmatched_top, multi_hit_top, kw_wide


# =========================
# 1) 上传数据文件
# =========================
st.subheader("1) 上传数据文件")
data_file = st.file_uploader("选择订单明细文件（Excel 或 CSV）", type=["xlsx", "xls", "csv"], key="data_file")
if data_file is None:
    st.stop()

with st.spinner("读取数据..."):
    raw_bytes = data_file.getvalue()
    df = read_data_file(data_file.name, raw_bytes)

df = compress_types(df)

# =========================
# 2) 上传规则文件（可选）
# =========================
st.subheader("2) 分类规则（可选：上传规则表 Excel/CSV）")
rule_file = st.file_uploader("上传分类规则表（两列：分类、关键词）", type=["xlsx", "xls", "csv"], key="rule_file")

rule_source = "默认规则（内置）"
rule_updated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if rule_file is not None:
    with st.spinner("读取规则表..."):
        b = rule_file.getvalue()
        if rule_file.name.lower().endswith(".csv"):
            rule_df = read_csv_safely_generic(b)
        else:
            rule_df = pd.read_excel(io.BytesIO(b), sheet_name=0)

        CATEGORY_MAP = parse_rule_table(rule_df)
        rule_source = f"上传规则：{rule_file.name}"
else:
    CATEGORY_MAP = DEFAULT_CATEGORY_MAP

rule_fingerprint = sha1_text(json.dumps(CATEGORY_MAP, ensure_ascii=False, sort_keys=True))

# =========================
# Sidebar 配置
# =========================
with st.sidebar:
    st.header("配置")
    grain = st.selectbox("时间颗粒（最小30分钟）", ["30分钟", "60分钟", "120分钟"], index=0)
    grain_min = {"30分钟": 30, "60分钟": 60, "120分钟": 120}[grain]

    st.subheader("规则信息")
    st.write(f"- 来源：{rule_source}")
    st.write(f"- 指纹：`{rule_fingerprint}`")
    st.write(f"- 生成时间：{rule_updated_at}")

    st.subheader("做法单加白名单")
    st.write("、".join(ADDON_WHITELIST))

# =========================
# 3) 查询区（先选时间再点查询）
# =========================
st.subheader("3) 查询条件（先选时间范围，再点击查询）")

min_dt = pd.to_datetime(df["创建时间"], errors="coerce").min()
max_dt = pd.to_datetime(df["创建时间"], errors="coerce").max()
if pd.isna(min_dt) or pd.isna(max_dt):
    min_dt = pd.Timestamp.today().normalize()
    max_dt = min_dt

if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "queried" not in st.session_state:
    st.session_state.queried = False

col1, colmid, col2, col3 = st.columns([3, 0.5, 3, 1.5])
with col1:
    start_date = st.date_input("开始日期", value=min_dt.date())
    start_time = st.time_input("开始时间", value=dt.time(0, 0))
with colmid:
    st.markdown("<div style='text-align:center; font-size:28px; padding-top:28px;'>~</div>", unsafe_allow_html=True)
with col2:
    end_date = st.date_input("结束日期", value=max_dt.date())
    end_time = st.time_input("结束时间", value=dt.time(23, 59))
with col3:
    st.write(""); st.write("")
    do_query = st.button("🔍 查询", type="primary")

start_dt = pd.Timestamp.combine(start_date, start_time)
end_dt = pd.Timestamp.combine(end_date, end_time)
if start_dt > end_dt:
    st.error("开始时间不能晚于结束时间。")
    st.stop()

if do_query:
    st.session_state.last_query = (start_dt, end_dt, grain_min, rule_fingerprint)
    st.session_state.queried = True

if not st.session_state.queried:
    st.info("请先选择下单时间范围，然后点击【查询】开始分析。")
    st.stop()

current_sig = (start_dt, end_dt, grain_min, rule_fingerprint)
if st.session_state.last_query != current_sig:
    st.warning("你已修改了查询条件/配置，但尚未点击【查询】；当前结果仍是上一次查询的结果。")

q_start_dt, q_end_dt, q_grain_min, _ = st.session_state.last_query

df_q = df[(df["创建时间"] >= q_start_dt) & (df["创建时间"] <= q_end_dt)].copy()
if df_q.empty:
    st.warning("该时间范围内没有数据，请调整时间范围后再查询。")
    st.stop()

df_q["时间段"] = build_time_bucket(df_q, q_grain_min)

# =========================
# 分类展开 + 覆盖率
# =========================
cat_df, unmatched_df, coverage_df = build_category_long_df_and_coverage(df_q, CATEGORY_MAP)
if cat_df.empty:
    st.warning("当前分类关键词未命中任何菜品名称，请先完善分类关键词配置。")
    st.stop()

# 口径：销量贡献 & 营业额贡献
cat_df["销量贡献"] = cat_df["菜品数量"].astype("int64")
cat_df["营业额贡献"] = cat_df["优惠后小计价格"].astype("float64")

# 蛋类（依据分类基础）：营业额=2×数量；销量=数量（命中即送一个）
egg_mask = cat_df["分类基础"].isin(EGG_CATEGORIES)
if egg_mask.any():
    cat_df.loc[egg_mask, "营业额贡献"] = (
        cat_df.loc[egg_mask, "分类基础"].map(FIXED_PRICE_CATEGORY).astype("float64")
        * cat_df.loc[egg_mask, "销量贡献"].astype("float64")
    )

# KPI
total_sales = int(cat_df["销量贡献"].sum())
total_rev = float(cat_df["营业额贡献"].sum())
num_categories = int(cat_df["分类"].nunique())
unmatched_rows = int(coverage_df["未命中行数"].iloc[0])
hit_rate = float(coverage_df["命中率"].iloc[0])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("总销量（分类双计数）", f"{total_sales:,}")
k2.metric("总营业额（含蛋类修正）", f"{total_rev:,.2f}")
k3.metric("展示分类数（含单加-）", f"{num_categories}")
k4.metric("未命中行数", f"{unmatched_rows:,}")
k5.metric("命中率", f"{hit_rate*100:.2f}%")

st.success(f"✅ 已筛选：{q_start_dt} ~ {q_end_dt}；原始行数 {len(df_q):,}；分类展开后 {len(cat_df):,}（双计数）")

# =========================
# 统计：时段×分类 销量/占比
# =========================
qty_time = (
    cat_df.groupby(["时间段", "分类"], as_index=False)["销量贡献"].sum()
    .rename(columns={"销量贡献": "销量"})
)
qty_time["占比"] = qty_time["销量"] / qty_time.groupby("时间段")["销量"].transform("sum")

# 各分类总销量（不分规格） + 占比
qty_cat_total = (
    cat_df.groupby("分类", as_index=False)["销量贡献"].sum()
    .rename(columns={"销量贡献": "总销量"})
    .sort_values("总销量", ascending=False)
)
qty_cat_total["占比"] = qty_cat_total["总销量"] / qty_cat_total["总销量"].sum()

# 分类×规格 销量/占比
qty_spec = (
    cat_df.groupby(["分类", "规格名称"], as_index=False)["销量贡献"].sum()
    .rename(columns={"销量贡献": "销量"})
)
qty_spec["占比"] = qty_spec["销量"] / qty_spec.groupby("分类")["销量"].transform("sum")

# 规格总计销量（跨分类）：标准拆两桶（菜品标准 vs 其他标准）
spec_name = cat_df["规格名称"].astype(str).str.strip()
is_standard = spec_name.eq("标准")
is_food_standard = (
    is_standard
    & (~cat_df["分类"].astype(str).str.startswith("单加-"))
    & (cat_df["分类基础"].astype(str).ne("其他类"))
)
cat_df["规格展示"] = np.where(
    is_standard,
    np.where(is_food_standard, "菜品标准", "其他标准"),
    spec_name
)

spec_total_full = (
    cat_df.groupby("规格展示", as_index=False)["销量贡献"].sum()
    .rename(columns={"规格展示": "规格名称", "销量贡献": "总销量"})
    .sort_values("总销量", ascending=False)
)
spec_total_full["占比"] = spec_total_full["总销量"] / spec_total_full["总销量"].sum()

# 分类营业额（按展示分类）
rev_cat = (
    cat_df.groupby("分类", as_index=False)["营业额贡献"].sum()
    .rename(columns={"营业额贡献": "营业额"})
    .sort_values("营业额", ascending=False)
)

# 做法单加（向量化）
addon_summary = compute_addon_summary_vectorized(cat_df, ADDON_WHITELIST)

# 打包盒数量：打包盒(菜品) + 打包(做法)
box_item_qty = df_q.loc[df_q["菜品名称"].astype(str).str.contains("打包盒", na=False), "菜品数量"].sum()
box_item_qty = int(box_item_qty) if pd.notna(box_item_qty) else 0
method_pack_qty = int(addon_summary.loc[addon_summary["单加项"] == "打包", "数量"].sum()) if not addon_summary.empty else 0
packaging_total_qty = box_item_qty + method_pack_qty
packaging_df = pd.DataFrame([
    {"项": "打包盒（菜品名称含打包盒）数量", "数量": box_item_qty},
    {"项": "打包（做法出现次数×菜品数量）数量", "数量": method_pack_qty},
    {"项": "打包盒合计数量（两者相加）", "数量": packaging_total_qty},
])

# =========================
# 4) 可视化分析
# =========================
st.divider()
st.header("4) 可视化分析")

# ✅ 你要的：指定“一天内时段窗口”销量占比（饼图）
st.subheader(f"所选时间范围内：指定时段窗口销量占比（{grain}，任意天数适用）")

wcol1, wcol2, wcol3 = st.columns([1.2, 1.2, 2])
with wcol1:
    window_start = st.time_input("窗口开始（一天内）", value=dt.time(11, 30), key="window_start")
with wcol2:
    window_end = st.time_input("窗口结束（一天内）", value=dt.time(14, 30), key="window_end")
with wcol3:
    st.caption("口径：在你已【查询】的日期范围内，统计落在该窗口内的销量占总销量的比例（窗口内 vs 窗口外）。")

range_total = float(cat_df["销量贡献"].sum())
if range_total <= 0:
    st.warning("当前查询范围内总销量为0，无法计算占比。")
else:
    tod = pd.to_datetime(cat_df["时间段"]).dt.time

    if window_start <= window_end:
        in_window = (tod >= window_start) & (tod < window_end)
        window_label = f"{window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}"
    else:
        in_window = (tod >= window_start) | (tod < window_end)
        window_label = f"{window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}（跨午夜）"

    window_sales = float(cat_df.loc[in_window, "销量贡献"].sum())
    other_sales = max(range_total - window_sales, 0.0)
    share = window_sales / range_total if range_total > 0 else 0.0

    kA, kB, kC = st.columns(3)
    kA.metric("查询范围总销量", f"{int(range_total):,}")
    kB.metric(f"窗口内销量（{window_label}）", f"{int(window_sales):,}")
    kC.metric("窗口占比", f"{share*100:.2f}%")

    pie = pd.DataFrame({
        "部分": [f"窗口内 {window_label}", "窗口外"],
        "销量": [window_sales, other_sales]
    })
    st.plotly_chart(px.pie(pie, names="部分", values="销量", hole=0.45), use_container_width=True)

    with st.expander("查看按日期拆分（窗口内/窗口外）", expanded=False):
        tmp = cat_df.copy()
        tmp["日期"] = pd.to_datetime(tmp["时间段"]).dt.date.astype(str)
        tmp["是否窗口内"] = in_window
        daily = (
            tmp.groupby(["日期", "是否窗口内"], as_index=False)["销量贡献"].sum()
            .rename(columns={"销量贡献": "销量"})
        )
        daily["部分"] = daily["是否窗口内"].map({True: "窗口内", False: "窗口外"})
        daily = daily.pivot(index="日期", columns="部分", values="销量").fillna(0).reset_index()
        if "窗口内" not in daily.columns:
            daily["窗口内"] = 0
        if "窗口外" not in daily.columns:
            daily["窗口外"] = 0
        daily["总计"] = daily["窗口内"] + daily["窗口外"]
        daily["窗口占比"] = daily["窗口内"] / daily["总计"].replace(0, np.nan)
        st.dataframe(daily, use_container_width=True)

# 其余可视化：营业额 Top20 + 时段分类堆叠
c1, c2 = st.columns(2)
with c1:
    st.subheader("各分类营业额（Top20，含单加-）")
    st.plotly_chart(px.bar(rev_cat.head(20), x="营业额", y="分类", orientation="h"), use_container_width=True)

with c2:
    st.subheader(f"各时间段各分类销量（{grain}，含单加-）")
    pivot = qty_time.pivot_table(index="时间段", columns="分类", values="销量", aggfunc="sum").fillna(0).reset_index()
    y_cols = [c for c in pivot.columns if c != "时间段"]
    st.plotly_chart(px.bar(pivot, x="时间段", y=y_cols), use_container_width=True)

st.subheader("各分类总销量（Top20，含单加-）")
st.plotly_chart(px.bar(qty_cat_total.head(20), x="总销量", y="分类", orientation="h"), use_container_width=True)

st.subheader("各分类总销量占比（Top20 + 其他）")
pie_df = make_topn_with_others(qty_cat_total[["分类", "总销量"]], "分类", "总销量", topn=20)
st.plotly_chart(px.pie(pie_df, names="分类", values="总销量", hole=0.45), use_container_width=True)

# =========================
# 规格总计销量
# =========================
st.subheader("规格总计销量（跨分类：宽面/细面/菜品标准/其他标准等）")
spec_col1, spec_col2, spec_col3 = st.columns([2, 1, 2])
with spec_col1:
    spec_topn = st.selectbox("显示 TopN", [10, 20, 50, 100, 999999], index=1)
with spec_col2:
    only_nonzero = st.checkbox("仅显示有销量", value=True)
with spec_col3:
    spec_search = st.text_input("规格搜索（关键字）", value="", placeholder="例如：米饭 / 宽面 / 细面 / 菜品标准")

spec_total = spec_total_full.copy()
if only_nonzero:
    spec_total = spec_total[spec_total["总销量"] > 0]
if spec_search.strip():
    kw = spec_search.strip()
    spec_total = spec_total[spec_total["规格名称"].astype(str).str.contains(re.escape(kw), na=False)]
if spec_topn != 999999:
    spec_total = spec_total.head(int(spec_topn))
st.dataframe(spec_total, use_container_width=True)

# =========================
# 明细表
# =========================
st.subheader("各分类总销量（不分规格）明细")
st.dataframe(qty_cat_total, use_container_width=True)

st.subheader("各时间段各分类销量与占比（明细，含单加-）")
st.dataframe(qty_time.sort_values(["时间段", "销量"], ascending=[True, False]), use_container_width=True)

st.subheader("各分类各规格销量与占比（含单加-；不会混入大类）")
st.dataframe(qty_spec.sort_values(["分类", "销量"], ascending=[True, False]), use_container_width=True)

st.subheader("做法单加项统计（出现次数 × 菜品数量）")
st.dataframe(addon_summary, use_container_width=True)

st.subheader("打包盒统计（含做法“打包”数量）")
st.dataframe(packaging_df, use_container_width=True)

# =========================
# 5) 规则体检
# =========================
st.divider()
st.header("5) 规则体检")

hc_col1, hc_col2 = st.columns([1, 3])
with hc_col1:
    do_health = st.button("🧪 运行规则体检", type="secondary")
with hc_col2:
    st.caption("输出：未命中Top、多分类命中Top（冲突）、关键词过宽Top（可能需要收缩/精确化）")

if do_health:
    with st.spinner("体检中..."):
        unmatched_top, multi_hit_top, kw_wide = rule_health_check(df_q, CATEGORY_MAP, unmatched_df, cat_df)

    st.subheader("未命中Top（建议补充关键词）")
    st.dataframe(unmatched_top, use_container_width=True)

    st.subheader("多分类命中Top（可能关键词过宽/冲突）")
    st.dataframe(multi_hit_top, use_container_width=True)

    st.subheader("关键词过宽Top（命中菜品数多的关键词）")
    st.dataframe(kw_wide, use_container_width=True)

# =========================
# 6) 抽样审计
# =========================
st.divider()
st.header("6) 抽样审计")

audit_col1, audit_col2, audit_col3 = st.columns([1, 1, 2])
with audit_col1:
    audit_n = st.selectbox("抽样行数", [10, 20, 50, 100], index=1)
with audit_col2:
    audit_seed = st.number_input("随机种子", min_value=0, max_value=999999, value=42, step=1)
with audit_col3:
    st.caption("随机抽原始行 → 展示命中哪些分类/是否单加/是否蛋/时间桶，方便快速核对规则与口径。")

rng = np.random.default_rng(int(audit_seed))
if len(df_q) > 0:
    sample_idx = rng.choice(df_q.index.to_numpy(), size=min(int(audit_n), len(df_q)), replace=False)
    audit_raw = df_q.loc[sample_idx, ["创建时间", "菜品名称", "菜品数量", "规格名称", "做法", "优惠后小计价格", "时间段"]].copy()

    key_cols = ["创建时间", "菜品名称", "菜品数量", "规格名称", "做法", "优惠后小计价格", "时间段"]
    audit_raw["_k"] = audit_raw[key_cols].astype(str).agg("|".join, axis=1)
    tmp = cat_df.copy()
    tmp["_k"] = tmp[key_cols].astype(str).agg("|".join, axis=1)

    hit = (
        tmp.groupby("_k", as_index=False)
        .agg(
            命中分类=("分类基础", lambda x: "，".join(sorted(set(map(str, x))))),
            展示分类=("分类", lambda x: "，".join(sorted(set(map(str, x))))),
        )
    )

    audit = audit_raw.merge(hit, on="_k", how="left")
    audit["是否单加"] = audit["菜品名称"].astype(str).str.strip().str.startswith("加")
    audit["是否蛋类"] = audit["命中分类"].astype(str).str.contains("卤蛋类|煎蛋类", na=False)
    audit = audit.drop(columns=["_k"])
    st.dataframe(audit, use_container_width=True)
else:
    st.info("当前时间范围内无数据，无法抽样审计。")

# =========================
# 7) 分类质量检查（覆盖率 + 未命中）
# =========================
st.divider()
st.header("7) 分类质量检查")

st.subheader("分类覆盖率")
st.dataframe(coverage_df, use_container_width=True)

st.subheader("未命中分类的菜品（建议补充关键词）")
unmatched_agg = (
    unmatched_df.groupby("菜品名称", as_index=False)
    .agg(
        未命中行数=("菜品名称", "size"),
        数量合计=("菜品数量", "sum"),
        小计合计=("优惠后小计价格", "sum")
    )
    .sort_values(["数量合计", "小计合计"], ascending=False)
)
st.dataframe(unmatched_agg, use_container_width=True)

# =========================
# 8) 导出 Excel（多 sheet）
# =========================
@st.cache_data(show_spinner=False)
def export_excel(
    qty_time, qty_cat_total, qty_spec, spec_total_full, rev_cat,
    addon_summary, packaging_df, coverage_df, unmatched_agg,
    unmatched_df, rule_fingerprint, rule_source,
    window_start: dt.time, window_end: dt.time
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        meta = pd.DataFrame([
            {"键": "规则来源", "值": rule_source},
            {"键": "规则指纹", "值": rule_fingerprint},
            {"键": "导出时间", "值": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"键": "窗口开始(一天内)", "值": window_start.strftime("%H:%M")},
            {"键": "窗口结束(一天内)", "值": window_end.strftime("%H:%M")},
        ])
        meta.to_excel(writer, sheet_name="元信息", index=False)

        qty_cat_total.to_excel(writer, sheet_name="分类总销量", index=False)
        qty_time.to_excel(writer, sheet_name="时段_分类销量占比", index=False)
        qty_spec.to_excel(writer, sheet_name="分类_规格销量占比", index=False)
        spec_total_full.to_excel(writer, sheet_name="规格总销量_全量", index=False)
        rev_cat.to_excel(writer, sheet_name="分类营业额", index=False)
        addon_summary.to_excel(writer, sheet_name="做法单加项统计", index=False)
        packaging_df.to_excel(writer, sheet_name="打包盒统计", index=False)
        coverage_df.to_excel(writer, sheet_name="分类覆盖率", index=False)
        unmatched_agg.to_excel(writer, sheet_name="未命中菜品汇总", index=False)
        unmatched_df.to_excel(writer, sheet_name="未命中明细", index=False)
    return output.getvalue()

st.divider()
st.header("8) 导出结果")

xlsx_bytes = export_excel(
    qty_time, qty_cat_total, qty_spec, spec_total_full, rev_cat,
    addon_summary, packaging_df, coverage_df, unmatched_agg,
    unmatched_df, rule_fingerprint, rule_source,
    window_start, window_end
)
st.download_button(
    label="下载统计结果（Excel）",
    data=xlsx_bytes,
    file_name="菜品销售分析.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
