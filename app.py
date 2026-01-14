import io
import inspect
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit

# -----------------------------
# Models
# -----------------------------
def singleCptNonVasc(time, a, b, v):
    return v * np.exp(-a * time) * np.sin(2 * np.pi * b * time)

def twoCptVasc(time, a, b, u, v):
    return 2 * np.exp(-a * time) * (u * np.cos(2 * np.pi * b * time) + v * np.sin(2 * np.pi * b * time))

def twoCptNonVascNoEH(time, M, a, b, u1, v1, ka):
    return M * np.exp(-ka * time) + 2 * np.exp(-a * time) * (u1 * np.cos(2 * np.pi * b * time) + v1 * np.sin(2 * np.pi * b * time))

def twoCptNonVascWithEH(time, M1, a, b, u, v, ka1):
    return M1 * np.exp(-ka1 * time) + 2 * np.exp(-a * time) * (u * np.cos(2 * np.pi * b * time) + v * np.sin(2 * np.pi * b * time))

def threeCptVasc(time, M, a, b, u, v, pai):
    return M * np.exp(-pai * time) + 2 * np.exp(-a * time) * (u * np.cos(2 * np.pi * b * time) + v * np.sin(2 * np.pi * b * time))

def threeCptNonVascNoEH(time, a, b, u1, v1, ka, pai, M1, M2):
    return M1 * np.exp(-ka * time) + M2 * np.exp(-pai * time) + 2 * np.exp(-a * time) * (u1 * np.cos(2 * np.pi * b * time) + v1 * np.sin(2 * np.pi * b * time))

def threeCptNonVascEHOneCplx(time, a, b, u, v, ka, pai, Ma1, M11):
    return Ma1 * np.exp(-ka * time) + M11 * np.exp(-pai * time) + 2 * np.exp(-a * time) * (u * np.cos(2 * np.pi * b * time) + v * np.sin(2 * np.pi * b * time))

def threeCptNonVascEHTwoCplx(time, a, b, u, v, d, g, x, y):
    return (
        2 * np.exp(-d * time) * (x * np.cos(2 * np.pi * g * time) + y * np.sin(2 * np.pi * g * time)) +
        2 * np.exp(-a * time) * (u * np.cos(2 * np.pi * b * time) + v * np.sin(2 * np.pi * b * time))
    )

# -----------------------------
# Fit + metrics
# -----------------------------
def fit_model(model, time, conc, initial_guess, maxfev=200000):
    params, _ = curve_fit(model, time, conc, p0=initial_guess, maxfev=maxfev)
    pred = model(time, *params)
    return params, pred

def r_squared(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def cmax_tmax_observed(time, conc):
    idx = int(np.argmax(conc))
    return float(conc[idx]), float(time[idx])

def cmax_tmax_fitted(tgrid, yfit):
    idx = int(np.argmax(yfit))
    return float(yfit[idx]), float(tgrid[idx])

def half_life_from_curve(tgrid, yfit):
    """
    用拟合曲线的“末端对数线性段”估计 t1/2（更通用，适配你的振荡/多峰模型）
    策略：
      - 取末端 40% 时间点
      - 仅保留 y > 0 的点（避免 log 无意义）
      - 对 ln(y) ~ t 做线性回归，斜率 = -k，t1/2 = ln(2)/k
    若末端不满足单调衰减或正值点不足，返回 NaN。
    """
    n = len(tgrid)
    start = int(np.floor(n * 0.60))
    t_tail = tgrid[start:]
    y_tail = yfit[start:]

    mask = np.isfinite(y_tail) & (y_tail > 0) & np.isfinite(t_tail)
    t_tail = t_tail[mask]
    y_tail = y_tail[mask]

    if len(t_tail) < 3:
        return float("nan")

    ln_y = np.log(y_tail)
    # ln(y) = c + m*t
    m, c = np.polyfit(t_tail, ln_y, 1)

    # m 应为负；k = -m
    k = -m
    if not np.isfinite(k) or k <= 0:
        return float("nan")

    return float(np.log(2) / k)

# -----------------------------
# Parameter calculators
# -----------------------------
def params_singleCptNonVasc(a, b, v, X0, Vc):
    b = 2 * np.pi * b
    AUC = (v * b) / (a**2 + b**2)
    k = X0 / (AUC * Vc)
    ka1 = (a**2 + b**2) / k
    k1a = 2 * a - k - ka1
    return {"AUC": AUC, "k": k, "ka1": ka1, "k1a": k1a}

def params_twoCptVasc(a, b, u, v, X0, Vc):
    b = 2 * np.pi * b
    AUC = (2*u*a + 2*v*b)/(a**2 + b**2)
    k = (X0 * 1000) / (AUC * Vc)
    k21 = (a**2 + b**2) / k
    k12 = 2 * a - k - k21
    return {"k": k, "k21": k21, "k12": k12, "AUC": AUC}

def params_twoCptNonVascNoEH(M, a, b, u1, v1, ka, X0, Vc):
    b = 2 * np.pi * b
    AUC = M/ka + (2*u1*a + 2*v1*b)/(a**2 + b**2)
    u = (u1 * (ka - a) + v1 * b) / ka
    v = (v1 * (ka - a) - u1 * b) / ka
    k21 = (b * v) / u + a
    k = (X0 * 1000)/(Vc * AUC)
    k12 = 2 * a - k - k21
    return {"k21": k21, "k": k, "k12": k12, "AUC": AUC, "ka": ka}

def params_twoCptNonVascWithEH(M1, a, b, u, v, ka1, X0, Vc):
    b = 2 * np.pi * b
    AUC = M1 / ka1 + (2*u*a + 2*v*b)/ (a**2 + b**2)
    k = X0 * 1000 / (Vc * AUC)
    h2 = (b * (ka1 - a) * v - b**2 * u) / ((ka1 - a) * u + b * v)
    k21 = h2 + a
    ka1 = (ka1 * (a**2 + b**2)) / (k * k21)
    f1 = 2 * a + ka1
    f2 = ka1 * 2 * a + a**2 + b**2
    k1a = (ka1**2 - f1 * ka1 + f2 - k * k21) / (k21 - ka1)
    k12 = (f1 * k21 - k21**2 + k * ka1 - f2) / (k21 - ka1)
    return {"k": k, "ka1": ka1, "k1a": k1a, "k21": k21, "k12": k12, "AUC": AUC, "h2": h2, "f1": f1, "f2": f2}

def _quad_roots(sum_, prod_):
    disc = sum_**2 - 4*prod_
    if np.any(disc < 0):
        disc = np.sqrt(disc.astype(complex))
    else:
        disc = np.sqrt(disc)
    h2 = (sum_ + disc)/2
    h3 = (sum_ - disc)/2
    return h2, h3

def params_threeCptVasc(M, a, b, u, v, pai, X0, Vc):
    b = 2 * np.pi * b
    AUC = M/pai + (2*u*a + 2*v*b)/(a**2+b**2)
    hpai = pai - a

    A = 2*u + M
    B = -2*u*hpai - M*hpai
    C = M*b*b - 2*u*hpai*hpai
    D = u*hpai + b*v
    E = u*b*b - hpai*b*v
    F = b*b*(u*hpai + b*v)

    h2ch3 = (C * E - B * F) / (A * E - B * D)
    h2jh3 = (A * F - C * D) / (A * E - B * D)
    h2, h3 = _quad_roots(h2jh3, h2ch3)

    k31 = h3 + a
    k21 = h2 + a
    k = X0*1000 / (Vc * AUC)

    f3 = 2 * a + pai - k - k21 - k31
    f4 = 2 * a * pai + a**2 + b**2 - k * (k21 + k31) - k21 * k31

    k12 = (f3*k21 - f4) / (k21 - k31)
    k13 = (f4 - f3 * k31) / (k21 - k31)

    return {"k": k, "k12": k12, "k13": k13, "k21": k21, "k31": k31, "AUC": AUC}

def params_threeCptNonVascNoEH(a, b, u1, v1, ka, pai, M1, M2, X0, Vc):
    b = 2 * np.pi * b
    AUC = M1 / ka + M2 / pai + 2 * u1 * a / (a**2 + b**2) + 2 * v1 * b / (a**2 + b**2)

    ha = ka - a
    hpai = pai - a
    u = (u1 * (ka - a) + v1 * b) / ka
    v = (v1 * (ka - a) - u1 * b) / ka
    f1 = (-(ha**2 + b**2) * M1) / ((hpai**2 + b**2) * M2)

    A = 1 - f1
    B = -f1 * (ha - hpai)
    C = f1 * (hpai**2) - (ha**2)
    D = u * hpai + b * v
    E = u * (b**2) - hpai * b * v
    F = b**2 * (u * hpai + b * v)

    h2ch3 = (C * E - B * F) / (A * E - B * D)
    h2jh3 = (A * F - C * D) / (A * E - B * D)
    h2, h3 = _quad_roots(h2jh3, h2ch3)

    k31 = h3 + a
    k21 = h2 + a
    k = X0 * 1000 / (Vc * AUC)

    f3 = 2 * a + pai - k - k21 - k31
    f4 = 2 * a * pai + a**2 + b**2 - k * (k21 + k31) - k21 * k31

    k12 = (f3 * k21 - f4) / (k21 - k31)
    k13 = (f4 - f3 * k31) / (k21 - k31)

    return {"k": k, "AUC": AUC, "ka": ka, "k12": k12, "k13": k13, "k21": k21, "k31": k31}

def params_threeCptNonVascEHOneCplx(a, b, u, v, ka, pai, Ma1, M11, X0, Vc):
    b = 2 * np.pi * b
    AUC = Ma1 / ka + M11 / pai + 2 * u * a / (a**2 + b**2) + 2 * v * b / (a**2 + b**2)

    ha = ka - a
    hpai = pai - a
    f1 = (-Ma1 * (ha**2 + b**2)) / ((hpai**2 + b**2) * M11)

    A = v * b * (hpai + ha) - u * (ha * hpai - (b**2))
    B = u * (b**2) * (hpai + ha) - v * b * (ha * hpai - (b**2))
    C = v * (b**3) * (hpai + ha) + u * (b**2) * (ha * hpai - (b**2))
    D = 1 - f1
    E = -(ha - hpai * f1)
    F = f1 * (hpai**2) - (ha**2)

    h2ch3 = (C * E - B * F) / (A * E - B * D)
    h2jh3 = (A * F - C * D) / (A * E - B * D)
    h2, h3 = _quad_roots(h2jh3, h2ch3)

    k31 = h3 + a
    k21 = h2 + a
    k = X0 * 1000 / (Vc * AUC)

    ka1 = ((a**2 + b**2) * pai * ka) / (k * k21 * k31)

    f2 = 2 * a + pai + ka - (k + k21 + k31 + ka1)
    f3 = a**2 + b**2 + 2 * a * pai + (2 * a + pai) * ka - k * k21 - k * k31 - k * ka1 - k21 * k31 - k21 * ka1 - k31 * ka1
    f4 = (a**2 + b**2) * pai + (a**2 + b**2 + 2 * a * pai) * ka - k * k21 * k31 - k * k21 * ka1 - k * ka1 * k31 - ka1 * k21 * k31
    f5 = k21 + k31
    f6 = ka1 + k31
    f7 = ka1 + k21
    f8 = k21 * k31
    f9 = ka1 * k31
    f10 = ka1 * k21

    denom = (f6 * f10 + f5 * f9 + f7 * f8 - f6 * f8 - f7 * f9 - f5 * f10)

    k1a = (f2 * f6 * f10 + f3 * f9 + f7 * f4 - f4 * f6 - f2 * f7 * f9 - f10 * f3) / denom
    k12 = (f3 * f10 + f4 * f5 + f2 * f7 * f8 - f3 * f8 - f2 * f5 * f10 - f4 * f7) / denom
    k13 = (f4 * f6 + f2 * f9 * f5 + f3 * f8 - f2 * f6 * f8 - f3 * f9 - f4 * f5) / denom

    return {"k": k, "AUC": AUC, "k12": k12, "k13": k13, "k21": k21, "k31": k31, "ka1": ka1, "k1a": k1a}

def params_threeCptNonVascEHTwoCplx(a, b, u, v, d, g, x, y, X0, Vc):
    b = 2 * np.pi * b
    g = 2 * np.pi * g

    AUC = 2 * (u * a + v * b) / (a**2 + b**2) + 2 * (x * d + y * g) / (d**2 + g**2)

    had = a - d
    hbg_minus = b - g
    hbg_plus = b + g

    q1 = (-had * hbg_minus) - had * hbg_plus
    w1 = had**2 - hbg_minus * hbg_plus
    q2 = -had * hbg_minus + had * hbg_plus
    w2 = had**2 + hbg_minus * hbg_plus

    A = u * w1 + q1 * v
    B = -(w1 * v * b - q1 * u * b)
    C = b**2 * (u * w1 + q1 * v)
    D = y * q2 + x * w2
    E = -w2 * y * g + q2 * x * g + (y * q2 + x * w2) * had
    F = 2 * had * (w2 * y * g - q2 * x * g) - (y * q2 + x * w2) * (had**2 - g**2)

    h2ch3 = (C * E - B * F) / (A * E - B * D)
    h2jh3 = (A * F - C * D) / (A * E - B * D)
    h2, h3 = _quad_roots(h2jh3, h2ch3)

    k31 = h3 + a
    k21 = h2 + a

    k = X0 * 1000 / (Vc * AUC)
    ka1 = ((a**2 + b**2) * (d**2 + g**2)) / (k * k21 * k31)

    f2 = 2 * a + 2 * d - (k + k21 + k31 + ka1)
    f3 = a**2 + b**2 + d**2 + g**2 + 4 * a * d - k * k21 - k * k31 - k * ka1 - k21 * k31 - k21 * ka1 - k31 * ka1
    f4 = (a**2 + b**2) * 2 * d + (d**2 + g**2) * 2 * a - k * k21 * k31 - k * k21 * ka1 - k * ka1 * k31 - ka1 * k21 * k31
    f5 = k21 + k31
    f6 = ka1 + k31
    f7 = ka1 + k21
    f8 = k21 * k31
    f9 = ka1 * k31
    f10 = ka1 * k21

    denom = (f6 * f10 + f5 * f9 + f7 * f8 - f6 * f8 - f7 * f9 - f5 * f10)

    k1a = (f2 * f6 * f10 + f3 * f9 + f7 * f4 - f4 * f6 - f2 * f7 * f9 - f10 * f3) / denom
    k12 = (f3 * f10 + f4 * f5 + f2 * f7 * f8 - f3 * f8 - f2 * f5 * f10 - f4 * f7) / denom
    k13 = (f4 * f6 + f2 * f9 * f5 + f3 * f8 - f2 * f6 * f8 - f3 * f9 - f4 * f5) / denom

    return {"k": k, "AUC": AUC, "k12": k12, "k13": k13, "k21": k21, "k31": k31, "ka1": ka1, "k1a": k1a}

MODELS = {
    "一室非血管模型": (singleCptNonVasc, params_singleCptNonVasc),
    "二室血管模型": (twoCptVasc, params_twoCptVasc),
    "二室非血管无消化道循环模型": (twoCptNonVascNoEH, params_twoCptNonVascNoEH),
    "二室非血管有消化道循环模型": (twoCptNonVascWithEH, params_twoCptNonVascWithEH),
    "三室血管模型": (threeCptVasc, params_threeCptVasc),
    "三室非血管无消化道循环模型": (threeCptNonVascNoEH, params_threeCptNonVascNoEH),
    "三室非血管消化道循环一对复数根模型": (threeCptNonVascEHOneCplx, params_threeCptNonVascEHOneCplx),
    "三室非血管消化道循环两对复数根模型": (threeCptNonVascEHTwoCplx, params_threeCptNonVascEHTwoCplx),
}

DEFAULT_GUESS = {
    "a": 0.8, "b": 0.5, "u": 0.3, "v": 0.4, "u1": 0.3, "v1": 0.4,
    "d": 0.3, "g": 0.1, "x": 1.0, "y": 0.5,
    "M": 1.0, "M1": 1.5, "M2": 1.0, "Ma1": 1.5, "M11": 0.8,
    "pai": 1.2, "ka": 1.2, "ka1": 1.2
}

def param_names(func):
    sig = inspect.signature(func)
    return [p.name for p in list(sig.parameters.values())[1:]]

def read_excel_any(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "time" in lower:
        tcol = lower["time"]
        time = pd.to_numeric(df[tcol], errors="coerce").values
        df = df.drop(columns=[tcol])
    else:
        time = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        df = df.iloc[:, 1:]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 0:
        raise ValueError("未检测到数值型浓度列。请确保表格包含至少一列数值型浓度数据。")
    return time, df, num_cols

def make_template():
    df = pd.DataFrame({"time": [0, 0.083, 0.25, 0.5, 1, 2, 4, 6, 8], "conc": [0, 10, 25, 20, 15, 8, 4, 2, 1]})
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="data")
    return bio.getvalue()

def safe_float(x):
    try:
        if isinstance(x, complex):
            return float(np.real(x))
        return float(x)
    except Exception:
        return float("nan")

def best_by_r2(df_metrics: pd.DataFrame):
    r2 = df_metrics["R2"].copy()
    r2 = r2.where(np.isfinite(r2), -np.inf)
    if len(r2) == 0:
        return None
    return int(np.argmax(r2.values))

def sheet_name_safe(name: str, max_len: int = 24):
    s = "".join([c if c.isalnum() else "_" for c in name]).strip("_")
    return s[:max_len] if s else "model"

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="PKparmas Web", page_icon="🧪", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: 0.2px;}
.pk-subtle {color: rgba(49,51,63,0.75); font-size: 0.95rem;}
.pk-card {border: 1px solid rgba(49,51,63,0.10); border-radius: 18px; padding: 16px; background: rgba(255,255,255,0.65);}
.pk-best {border: 2px solid rgba(49,51,63,0.18); box-shadow: 0 4px 20px rgba(0,0,0,0.06);}
.pk-plot {border: none !important; border-radius: 0 !important; padding: 0 !important; background: transparent !important; box-shadow: none !important;}
.pk-plot-best {border: none !important; border-radius: 0 !important; padding: 0 !important; background: transparent !important; box-shadow: none !important;}

.stButton>button {border-radius: 12px; padding: 0.6rem 1rem;}
[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,0.15); border-radius: 14px; padding: 10px 12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("PKparmas🧪")
st.caption("上传本地 Excel，系统对 8 种房室模型同时拟合并比较，以 R² 选择最优模型，并输出每个模型的拟合图与参数结果。")

with st.sidebar:
    st.header("输入")
    uploaded = st.file_uploader("上传 Excel（.xlsx）", type=["xlsx"])
    st.download_button("下载数据（xlsx）", data=make_template(), file_name="pkparmas_template.xlsx", use_container_width=True)

    st.markdown("---")
    st.subheader("全局参数")
    X0 = st.number_input("初始给药量 X0 (mg/kg)", value=0.67, min_value=0.0, step=0.01)
    Vc = st.number_input("表观分布容积 Vc (mL/kg)", value=270.0, min_value=0.0, step=1.0)
    maxfev = st.number_input("最大迭代次数", value=200000, min_value=20000, step=10000)

    st.markdown("---")
    st.subheader("初值设置（可选）")
    st.caption("默认提供一组通用初值；如某些模型拟合失败，可在此对该模型参数初值进行微调。")
    use_defaults_only = st.toggle("仅使用默认初值", value=False)

    guesses = {}
    if not use_defaults_only:
        for mname, (mfunc, _) in MODELS.items():
            pns = param_names(mfunc)
            with st.expander(mname, expanded=False):
                guesses[mname] = {}
                for p in pns:
                    guesses[mname][p] = st.number_input(
                        f"{p}",
                        value=float(DEFAULT_GUESS.get(p, 0.1)),
                        step=0.01,
                        format="%.6f",
                        key=f"{mname}_{p}"
                    )

    run = st.button("开始对比拟合", type="primary", use_container_width=True)

if uploaded is None:
    st.info("请在左侧上传 .xlsx 文件。支持两种格式：\n- 含 `time` 列 + 浓度列（如 `conc`）\n- 第一列为时间，后续列为浓度")
    st.stop()

try:
    time, df_conc, num_cols = read_excel_any(uploaded)
    if np.any(pd.isna(time)):
        raise ValueError("时间列存在非数值或缺失值，请检查。")
except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()

top = st.container()
with top:
    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.subheader("数据预览")
        conc_col = st.selectbox("选择浓度列", num_cols, index=0)
        conc = pd.to_numeric(df_conc[conc_col], errors="coerce").values
        if np.any(pd.isna(conc)):
            st.error("浓度列存在非数值或缺失值，请检查。")
            st.stop()
        st.dataframe(pd.DataFrame({"time": time, conc_col: conc}), use_container_width=True, height=280)
    with c2:
        st.subheader("说明")
        st.markdown(
            """
            <div class="pk-card">
              <div class="pk-subtle">
                (1) 点击左侧 “开始对比拟合” 按键后，将对 8 个模型同时拟合并比较。<br/>
                (2) 以 R² 最高的模型作为“最优模型”；拟合失败的模型会被标记。<br/>
                (3) 输出包含：拟合图、对比指标表、最优模型的拟合参数与派生参数，并提供一键下载。
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if not run:
    st.stop()

# Fit all models
results = []
per_model = {}

for mname, (mfunc, cfunc) in MODELS.items():
    pns = param_names(mfunc)
    if use_defaults_only:
        p0 = [float(DEFAULT_GUESS.get(p, 0.1)) for p in pns]
    else:
        p0 = [float(guesses.get(mname, {}).get(p, DEFAULT_GUESS.get(p, 0.1))) for p in pns]

    try:
        params, pred = fit_model(mfunc, time, conc, p0, maxfev=int(maxfev))
        r2 = r_squared(conc, pred)
        _rmse = rmse(conc, pred)
        _mae = mae(conc, pred)
        derived = cfunc(*params, float(X0), float(Vc))
        derived_norm = {k: safe_float(v) for k, v in derived.items()}

        results.append({"Model": mname, "R2": r2, "RMSE": _rmse, "MAE": _mae, "Status": "OK"})
        per_model[mname] = {
            "params": params,
            "param_names": pns,
            "pred": pred,
            "derived": derived_norm,
            "error": None,
        }
    except Exception as e:
        results.append({"Model": mname, "R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "Status": "Failed"})
        per_model[mname] = {"params": None, "param_names": pns, "pred": None, "derived": None, "error": str(e)}

metrics = pd.DataFrame(results)
metrics_sorted = metrics.sort_values(by=["R2"], ascending=False, na_position="last").reset_index(drop=True)
best_idx = best_by_r2(metrics_sorted)
best_model = metrics_sorted.loc[best_idx, "Model"] if best_idx is not None else None

st.markdown("---")
st.subheader("模型对比结果（按 R² 排序）")
st.dataframe(metrics_sorted, use_container_width=True, height=320)

if best_model is None or per_model[best_model]["pred"] is None:
    st.error("所有模型均拟合失败。建议：\n- 调整初值（左侧高级设置）\n- 增大 maxfev\n- 检查数据是否存在异常/离群点")
    st.stop()

st.markdown("---")
st.subheader("最优模型（R² 最高）")
st.markdown(
    f"""
    <div class="pk-card pk-best">
      <div style="font-size:1.05rem;"><b>最优模型：</b>{best_model}</div>
      <div class="pk-subtle">判据：R²。</div>
    </div>
    """,
    unsafe_allow_html=True
)

best_row = metrics[metrics["Model"] == best_model].iloc[0]
m1, m2, m3, m4 = st.columns([1,1,1,1], gap="medium")
m1.metric("R²", f"{best_row['R2']:.4f}" if np.isfinite(best_row["R2"]) else "NA")
m2.metric("RMSE", f"{best_row['RMSE']:.4f}" if np.isfinite(best_row["RMSE"]) else "NA")
m3.metric("MAE", f"{best_row['MAE']:.4f}" if np.isfinite(best_row["MAE"]) else "NA")
m4.metric("状态", str(best_row["Status"]))

best_info = per_model[best_model]
fit_df_best = pd.DataFrame({"parameter": best_info["param_names"], "estimate": best_info["params"]})
fit_df_best["estimate"] = fit_df_best["estimate"].apply(safe_float)
der_df_best = pd.DataFrame({"parameter": list(best_info["derived"].keys()), "value": list(best_info["derived"].values())})

cA, cB = st.columns([1,1], gap="large")
with cA:
    st.markdown("**拟合参数（最优模型）**")
    st.dataframe(fit_df_best, use_container_width=True, height=260)
with cB:
    st.markdown("**计算得到的参数（最优模型）**")
    st.dataframe(der_df_best, use_container_width=True, height=260)

st.markdown("**最优模型拟合曲线**")
st.markdown(f"最优模型：**{best_model}**")

try:
    import matplotlib.pyplot as plt
    tgrid = np.linspace(float(np.min(time)), float(np.max(time)), 350)
    mfunc, _ = MODELS[best_model]
    tpred = mfunc(tgrid, *best_info["params"])

    # ---- PK params: Cmax / Tmax / t1/2 ----
    Cmax_obs, Tmax_obs = cmax_tmax_observed(time, conc)
    Cmax_fit, Tmax_fit = cmax_tmax_fitted(tgrid, tpred)
    t12_fit = half_life_from_curve(tgrid, tpred)

    st.markdown("**药动学参数（基于当前浓度列）**")
    p1, p2, p3 = st.columns(3, gap="medium")
    p1.metric("Cmax (Observed)", f"{Cmax_obs:.6g}")
    p2.metric("Tmax (Observed)", f"{Tmax_obs:.6g}")
    p3.metric("t1/2 (Fitted, terminal)", f"{t12_fit:.6g}" if np.isfinite(t12_fit) else "NA")

    p4, p5, _ = st.columns(3, gap="medium")
    p4.metric("Cmax (Fitted)", f"{Cmax_fit:.6g}")
    p5.metric("Tmax (Fitted)", f"{Tmax_fit:.6g}")

    fig = plt.figure()
    plt.scatter(time, conc, label="Observed")
    plt.plot(tgrid, tpred, label="Fitted")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    # plt.title(best_model)
    plt.legend()

    # st.markdown('<div class="pk-plot pk-plot-best">', unsafe_allow_html=True)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    # st.markdown("</div>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"最优模型绘图失败：{e}")

st.markdown("**结果下载**")
d1, d2 = st.columns([1,1], gap="medium")

# with d1:
#     bio = io.BytesIO()
#     best_curve = pd.DataFrame({"time": time, "observed": conc, "predicted": best_info["pred"]})
#     with pd.ExcelWriter(bio, engine="openpyxl") as w:
#         metrics_sorted.to_excel(w, index=False, sheet_name="model_compare")
#         fit_df_best.to_excel(w, index=False, sheet_name="fit_params_best")
#         der_df_best.to_excel(w, index=False, sheet_name="derived_best")
#         best_curve.to_excel(w, index=False, sheet_name="curve_best")
#     st.download_button(
#         "下载：最优模型结果（xlsx）",
#         data=bio.getvalue(),
#         file_name=f"pkparmas_best_{best_model}_{conc_col}.xlsx".replace("/", "_"),
#         use_container_width=True
#     )

with d1:
    bio = io.BytesIO()

    # ====== 1) curve sheet ======
    best_curve = pd.DataFrame({
        "time": time,
        "observed": conc,
        "predicted": best_info["pred"]
    })

    # ====== 2) PK summary sheet (Cmax/Tmax/t1/2) ======
    # 前提：在此代码块之前，你已经计算出了这些变量：
    # Cmax_obs, Tmax_obs, Cmax_fit, Tmax_fit, t12_fit
    pk_summary = pd.DataFrame({
        "Metric": ["Cmax_obs", "Tmax_obs", "Cmax_fit", "Tmax_fit", "t1/2_fit_terminal"],
        "Value":  [Cmax_obs,   Tmax_obs,   Cmax_fit,   Tmax_fit,   t12_fit],
    })

    # ====== 3) write excel ======
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        metrics_sorted.to_excel(w, index=False, sheet_name="model_compare")
        fit_df_best.to_excel(w, index=False, sheet_name="fit_params_best")
        der_df_best.to_excel(w, index=False, sheet_name="derived_best")
        best_curve.to_excel(w, index=False, sheet_name="curve_best")
        pk_summary.to_excel(w, index=False, sheet_name="pk_summary")

    st.download_button(
        "下载：最优模型结果（xlsx）",
        data=bio.getvalue(),
        file_name=f"pkparmas_best_{best_model}_{conc_col}.xlsx".replace("/", "_"),
        use_container_width=True
    )


with d2:
    bio_all = io.BytesIO()
    with pd.ExcelWriter(bio_all, engine="openpyxl") as w:
        metrics_sorted.to_excel(w, index=False, sheet_name="model_compare")
        for mname, info in per_model.items():
            base = sheet_name_safe(mname)
            if info["pred"] is None:
                pd.DataFrame({"message": [f"Fit failed: {info['error']}"]}).to_excel(w, index=False, sheet_name=f"{base}_fail")
                continue
            fit_df = pd.DataFrame({"parameter": info["param_names"], "estimate": [safe_float(x) for x in info["params"]]})
            der = pd.DataFrame({"parameter": list(info["derived"].keys()), "value": list(info["derived"].values())})
            curve = pd.DataFrame({"time": time, "observed": conc, "predicted": info["pred"]})
            fit_df.to_excel(w, index=False, sheet_name=f"{base}_fit")
            der.to_excel(w, index=False, sheet_name=f"{base}_der")
            curve.to_excel(w, index=False, sheet_name=f"{base}_curve")
    st.download_button(
        "下载：全部模型结果（xlsx）",
        data=bio_all.getvalue(),
        file_name=f"pkparmas_all_models_{conc_col}.xlsx".replace("/", "_"),
        use_container_width=True
    )

st.markdown("---")
st.subheader("全部模型拟合图")

import matplotlib.pyplot as plt

model_names = list(MODELS.keys())
cols = st.columns(4, gap="medium")

for i, mname in enumerate(model_names):
    # re-create a new row after every 4 items
    if i % 4 == 0 and i != 0:
        cols = st.columns(4, gap="medium")
    col = cols[i % 4]

    with col:
        info = per_model[mname]
        is_best = (mname == best_model)
        cls = "pk-plot pk-plot-best" if is_best else "pk-plot"

        # st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        st.markdown(f"**{mname}**")

        if info["pred"] is not None:
            r2_val = metrics.loc[metrics["Model"] == mname, "R2"].values[0]
            st.caption(f"Status: OK | R² = {r2_val:.4f}" if np.isfinite(r2_val) else "Status: OK | R² = NA")
        else:
            st.caption("Status: Failed")

        fig = plt.figure()
        plt.scatter(time, conc, label="Obs")
        if info["pred"] is not None:
            tgrid = np.linspace(float(np.min(time)), float(np.max(time)), 240)
            mfunc, _ = MODELS[mname]
            try:
                tpred = mfunc(tgrid, *info["params"])
                plt.plot(tgrid, tpred, label="Fit")
            except Exception:
                pass
        plt.xlabel("T")
        plt.ylabel("C")
        plt.legend(loc="best", fontsize=8)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        if info["pred"] is None:
            st.caption(f"Reason: {info['error']}")
        # st.markdown("</div>", unsafe_allow_html=True)
