import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# 0. 文件路径（确保和 app.py 同目录）
# =======================
MODEL_PATH = Path("RF.pkl")
ZPARAMS_PATH = Path("zscore_params.pkl")

# =======================
# 1. 加载模型与预处理参数（缓存）
# =======================
@st.cache_resource
def load_model_and_params():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH.resolve()}")
    if not ZPARAMS_PATH.exists():
        raise FileNotFoundError(f"找不到预处理参数文件：{ZPARAMS_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    zparams = joblib.load(ZPARAMS_PATH)

    offset = float(zparams.get("offset", 0.0))
    mean = zparams["mean"]
    std = zparams["std"]
    return model, offset, mean, std

try:
    model, offset, mean, std = load_model_and_params()
except Exception as e:
    st.error(f"加载模型/参数失败：{e}")
    st.stop()

# =======================
# 2. 页面标题
# =======================
st.title("ESCC Prediction System (RF Model)")
st.markdown("### 输入 **原始代谢物值**，系统将自动进行 **log2 + Z-score（训练组参数）** 并预测是否为食管鳞癌（ESCC）")

with st.expander("预处理说明", expanded=False):
    st.write(
        "- log2：使用 log2(x + offset)\n"
        "- Z-score：使用训练组 mean/std： (log2值 - mean) / std\n"
        "- 预测时不能重新计算 mean/std（避免信息泄漏）"
    )
    st.write(f"offset = {offset}")

# =======================
# 3. 输入特征（原始代谢物值）
# =======================
st.sidebar.header("输入代谢物原始值")
st.sidebar.subheader("Metabolites")

Asparagine = st.sidebar.number_input("Asparagine", value=1.0, format="%.6f")
Choline = st.sidebar.number_input("Choline", value=1.0, format="%.6f")
Glutamate = st.sidebar.number_input("Glutamate", value=1.0, format="%.6f")
Sarcosine = st.sidebar.number_input("Sarcosine", value=1.0, format="%.6f")

feature_names = ["Asparagine", "Choline", "Glutamate", "Sarcosine"]

raw_df = pd.DataFrame([{
    "Asparagine": Asparagine,
    "Choline": Choline,
    "Glutamate": Glutamate,
    "Sarcosine": Sarcosine
}])

# =======================
# 4. 预测按钮
# =======================
if st.button("开始预测"):

    # 检查 mean/std 是否包含 4 个代谢物
    missing = [c for c in feature_names if c not in mean.index or c not in std.index]
    if missing:
        st.error(
            "zscore_params.pkl 中缺少以下代谢物的 mean/std：\n"
            f"{missing}\n\n"
            "请确认训练组文件列名与这里完全一致。"
        )
        st.stop()

    # 检查 log2 是否可计算
    min_allowed = -offset + 1e-12
    if (raw_df[feature_names] <= min_allowed).any().any():
        st.error(
            f"存在 <= {-offset} 的输入值，会导致 log2(x + offset) 无法计算。\n"
            f"请确保每个代谢物满足：x > {-offset}（offset={offset}）。"
        )
        st.stop()

    # 1) log2
    log2_df = np.log2(raw_df[feature_names].astype(float) + offset)

    # 2) Z-score（训练组参数）
    z_df = (log2_df - mean[feature_names]) / std[feature_names]

    # 3) 输入模型
    input_values = z_df[feature_names].values

    # 预测
    pred = int(model.predict(input_values)[0])
    probas = model.predict_proba(input_values)[0]  # [P(0), P(1)]

    st.markdown(f"### 🩺 预测结果: {'ESCC' if pred == 1 else 'Non-ESCC'}")
    st.write(f"**预测概率:** Non-ESCC (0) = {probas[0]:.4f}, ESCC (1) = {probas[1]:.4f}")

    # 展示预处理值（便于核对）
    with st.expander("查看预处理后的数值（raw / log2 / z-score）", expanded=True):
        show_df = pd.concat(
            [
                raw_df[feature_names].rename(columns=lambda x: f"{x} (raw)"),
                log2_df.rename(columns=lambda x: f"{x} (log2)"),
                z_df.rename(columns=lambda x: f"{x} (zscore)")
            ],
            axis=1
        )
        st.dataframe(show_df)

    # 建议文本
    prob_escc = probas[1] * 100
    if pred == 1:
        st.info(f"模型预测为 **ESCC（1）**，概率约为 **{prob_escc:.2f}%**。建议结合内镜、病理及临床评估进一步诊断。")
    else:
        st.info(f"模型预测为 **Non-ESCC（0）**，ESCC 概率约为 **{prob_escc:.2f}%**。但该结果仅供参考，仍建议结合个人临床风险因素，在医生指导下进行定期随访或筛查。")

    # 可视化
    plt.figure(figsize=(6, 3))
    plt.barh(["Non-ESCC (0)", "ESCC (1)"], [probas[0], probas[1]],color=["#2E86C1", "#E74C3C"])
    plt.xlabel("Predicted probability")
    for i, v in enumerate(probas):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    plt.xlim(0, 1)
    plt.tight_layout()
    st.pyplot(plt)
