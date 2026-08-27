import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

st.set_page_config(page_title="Smart Predictor | AI Lab", page_icon="🌸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #0c1a0e 0%, #14532d 30%, #064e3b 70%, #0c1a0e 100%); }

.page-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #4ade80, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }

.concept-box {
    background: rgba(74,222,128,0.07);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    color: #bbf7d0; font-size: 0.9rem; line-height: 1.6;
}
.concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #4ade80; }

.prediction-card {
    background: rgba(255,255,255,0.05);
    border-radius: 24px; padding: 2rem;
    text-align: center;
    border: 2px solid;
    margin-bottom: 1rem;
}
.pred-species  { font-size: 2.2rem; font-weight: 900; margin: 0.5rem 0; }
.pred-conf     { color: #94a3b8; font-size: 0.9rem; }

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 800; }
.metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

.slider-label { color: #a7f3d0; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }

.stButton > button {
    background: linear-gradient(135deg, #15803d, #6d28d9) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
}
.stSlider [data-testid="stSlider"] { color: #4ade80 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & train model (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(iris.data, iris.target)

    return rf, dt, iris, acc

rf_model, dt_model, iris_data, accuracy = load_model()
SPECIES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
SPECIES_EMOJI = ["🌱", "🌷", "🌺"]
SPECIES_COLORS = ["#4ade80", "#a78bfa", "#f472b6"]
SPECIES_BORDER = ["rgba(74,222,128,0.5)", "rgba(167,139,250,0.5)", "rgba(244,114,182,0.5)"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🌸 Smart Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Adjust the flower measurements and let the AI predict the Iris species!</div>', unsafe_allow_html=True)

main_col, sidebar_col = st.columns([3, 2], gap="large")

# ── Sidebar: concepts ──────────────────────────────────────────────────────────
with sidebar_col:
    st.markdown("""
    <div class="concept-box">
        <div class="concept-title">🌿 The Iris Dataset</div>
        Famous in ML! It has <strong>150 flower measurements</strong> across 3 species.
        Scientists measured petals and sepals to classify them — and we train AI to do the same!
    </div>
    <div class="concept-box">
        <div class="concept-title">🌲 What is a Random Forest?</div>
        It builds many <strong>decision trees</strong> and takes a vote. Like asking 100 experts
        and going with the majority — that's why it's more accurate than one tree!
    </div>
    <div class="concept-box">
        <div class="concept-title">📏 Features & Labels</div>
        <strong>Features</strong> = inputs the AI uses (petal/sepal size)<br><br>
        <strong>Label</strong> = the answer (flower species)<br><br>
        In <em>supervised learning</em>, the AI learns from labeled examples.
    </div>
    <div class="concept-box">
        <div class="concept-title">🎯 Model Accuracy</div>
        This model has <strong>{:.1f}%</strong> accuracy on test data — meaning it correctly
        identifies the species most of the time on flowers it has never seen!
    </div>
    """.format(accuracy * 100), unsafe_allow_html=True)

    # Dataset preview
    with st.expander("📊 View Raw Dataset (first 10 rows)", expanded=False):
        df = pd.DataFrame(iris_data.data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        df["Species"] = [SPECIES[t] for t in iris_data.target]
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# ── Main: Sliders + Prediction ─────────────────────────────────────────────────
with main_col:
    st.markdown("#### 🎛️ Adjust Flower Measurements")
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem">
    Move the sliders to set the measurements (in centimetres) and watch the AI predict in real-time!
    </div>
    """, unsafe_allow_html=True)

    sl_col1, sl_col2 = st.columns(2, gap="medium")

    with sl_col1:
        st.markdown('<div class="slider-label">📏 Sepal Length (cm)</div>', unsafe_allow_html=True)
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1, key="sl", label_visibility="collapsed")

        st.markdown('<div class="slider-label">📏 Sepal Width (cm)</div>', unsafe_allow_html=True)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1, key="sw", label_visibility="collapsed")

    with sl_col2:
        st.markdown('<div class="slider-label">🌿 Petal Length (cm)</div>', unsafe_allow_html=True)
        petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0, 0.1, key="pl", label_visibility="collapsed")

        st.markdown('<div class="slider-label">🌿 Petal Width (cm)</div>', unsafe_allow_html=True)
        petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3, 0.1, key="pw", label_visibility="collapsed")

    # Real-time prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = rf_model.predict(features)[0]
    pred_proba = rf_model.predict_proba(features)[0]
    confidence = pred_proba[pred_class] * 100

    species_name = SPECIES[pred_class]
    species_emoji = SPECIES_EMOJI[pred_class]
    pred_color = SPECIES_COLORS[pred_class]
    pred_border = SPECIES_BORDER[pred_class]

    st.markdown(f"""
    <div class="prediction-card" style="border-color: {pred_border}; background: rgba(0,0,0,0.2)">
        <div style="font-size:4rem">{species_emoji}</div>
        <div class="pred-species" style="color: {pred_color}">{species_name}</div>
        <div class="pred-conf">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars for all 3 species
    prob_col1, prob_col2, prob_col3 = st.columns(3, gap="small")
    for col, i in zip([prob_col1, prob_col2, prob_col3], range(3)):
        prob_pct = pred_proba[i] * 100
        is_top = (i == pred_class)
        with col:
            st.markdown(f"""
            <div class="metric-box" style="border-color:{'rgba(255,255,255,0.3)' if is_top else 'rgba(255,255,255,0.08)'}">
                <div style="font-size:1.5rem">{SPECIES_EMOJI[i]}</div>
                <div class="metric-val" style="color:{SPECIES_COLORS[i]}">{prob_pct:.0f}%</div>
                <div class="metric-lbl">{SPECIES[i].split()[-1]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance chart
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🌐 Dataset Scatter", "🌲 Decision Tree"])

    with tab1:
        importances = rf_model.feature_importances_
        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        sorted_idx = np.argsort(importances)

        fig_imp = go.Figure(go.Bar(
            x=importances[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation="h",
            marker=dict(
                color=["#a78bfa", "#60a5fa", "#4ade80", "#34d399"][:len(sorted_idx)],
                line=dict(width=0),
            ),
            text=[f"{v*100:.1f}%" for v in importances[sorted_idx]],
            textposition="auto",
            textfont=dict(color="white", size=13),
        ))
        fig_imp.update_layout(
            title=dict(text="Which features matter most to the AI?", font=dict(color="#e2e8f0", size=14)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#94a3b8", family="Outfit"),
            xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
            margin=dict(l=10, r=10, t=40, b=30),
            height=250,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.info("💡 **Petal measurements** matter much more than sepal measurements for this classification!")

    with tab2:
        df_plot = pd.DataFrame(iris_data.data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        df_plot["Species"] = [SPECIES[t] for t in iris_data.target]

        fig_scatter = px.scatter(
            df_plot, x="Petal Length", y="Petal Width", color="Species",
            color_discrete_map={s: c for s, c in zip(SPECIES, SPECIES_COLORS)},
            title="Iris Dataset: Petal Length vs Petal Width",
            labels={"Petal Length": "Petal Length (cm)", "Petal Width": "Petal Width (cm)"},
        )
        # Mark the current slider input
        fig_scatter.add_trace(go.Scatter(
            x=[petal_length], y=[petal_width],
            mode="markers",
            marker=dict(symbol="star", size=20, color=pred_color, line=dict(color="white", width=2)),
            name="Your Flower ⭐",
        ))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#94a3b8", family="Outfit"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            height=320,
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("⭐ The star shows where YOUR flower measurements fall in the dataset!")

    with tab3:
        st.markdown("#### 🌲 Simplified Decision Tree (depth 3)")
        tree_text = export_text(dt_model, feature_names=["Sepal Len", "Sepal Wid", "Petal Len", "Petal Wid"])
        st.code(tree_text, language="")
        st.info("💡 This tree shows the if-else rules the AI learned. The Random Forest uses 100 such trees!")
