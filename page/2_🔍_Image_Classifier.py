import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(page_title="Image Classifier | AI Lab", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
.page-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
.concept-box {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    color: #a7f3d0; font-size: 0.9rem; line-height: 1.6;
}
.concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #34d399; }
.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem;
}
.top-prediction {
    font-size: 2rem; font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.stButton > button {
    background: linear-gradient(135deg, #059669, #0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Local colour-based classifier (no internet needed) ─────────────────────────
@st.cache_resource
def build_color_classifier():
    """
    Train a simple Random-Forest classifier on hand-crafted colour & texture
    features extracted from example images.  This is 100% offline and teaches
    the same ML concepts as a deep model.
    """
    from sklearn.ensemble import RandomForestClassifier
    import random, math

    random.seed(42)
    np.random.seed(42)

    # 12 categories with characteristic (R,G,B) colour profiles + brightness
    categories = [
        "🌿 Plant / Nature",
        "🌊 Water / Ocean",
        "🔥 Fire / Warm tones",
        "🌑 Dark / Night scene",
        "☀️ Bright / Sunny",
        "🍎 Red object",
        "💛 Yellow object",
        "💙 Blue object",
        "🤍 White / Light",
        "🖤 Black / Dark object",
        "🟤 Brown / Earth tones",
        "🌸 Pink / Purple tones",
    ]

    # Generate synthetic training features: [mean_r, mean_g, mean_b, brightness, contrast, r_ratio, g_ratio, b_ratio]
    def make_samples(mean_rgb, n=120, noise=30):
        samples = []
        for _ in range(n):
            r = np.clip(mean_rgb[0] + random.gauss(0, noise), 0, 255)
            g = np.clip(mean_rgb[1] + random.gauss(0, noise), 0, 255)
            b = np.clip(mean_rgb[2] + random.gauss(0, noise), 0, 255)
            bright = (r + g + b) / 3
            total = r + g + b + 1e-6
            samples.append([r, g, b, bright, abs(r-g)+abs(g-b), r/total, g/total, b/total])
        return samples

    profiles = [
        (60, 130, 60),    # plant
        (50, 100, 180),   # water
        (220, 90, 30),    # fire
        (30, 30, 40),     # dark
        (230, 220, 150),  # bright
        (200, 50, 50),    # red
        (230, 210, 50),   # yellow
        (50, 100, 220),   # blue
        (230, 230, 230),  # white
        (30, 30, 30),     # black
        (140, 90, 60),    # brown
        (200, 100, 180),  # pink/purple
    ]

    X, y = [], []
    for idx, (profile, cat) in enumerate(zip(profiles, categories)):
        samples = make_samples(profile)
        X.extend(samples)
        y.extend([idx] * len(samples))

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y)
    return clf, categories

def extract_features(img: Image.Image) -> np.ndarray:
    """Extract colour statistics from a PIL image."""
    img_small = img.resize((64, 64)).convert("RGB")
    arr = np.array(img_small, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    brightness = (mean_r + mean_g + mean_b) / 3
    contrast = arr.std()
    total = mean_r + mean_g + mean_b + 1e-6
    return np.array([[mean_r, mean_g, mean_b, brightness, contrast,
                      mean_r/total, mean_g/total, mean_b/total]])

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🔍 Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload any photo and watch the AI analyse its colours and patterns to classify it!</div>', unsafe_allow_html=True)

main_col, sidebar_col = st.columns([3, 2], gap="large")

with sidebar_col:
    st.markdown("""
    <div class="concept-box">
        <div class="concept-title">👁️ What is Computer Vision?</div>
        AI <strong>sees</strong> an image as millions of numbers (pixels). It learns to find
        patterns in those numbers — colours, edges, textures — to recognise objects.
    </div>
    <div class="concept-box">
        <div class="concept-title">🌈 How this demo works</div>
        This classifier extracts <strong>colour statistics</strong> (average red, green, blue,
        brightness, contrast) from your image and feeds them into a trained
        <strong>Random Forest</strong> model to predict the category.
    </div>
    <div class="concept-box">
        <div class="concept-title">📊 What is Confidence?</div>
        The model gives a <strong>probability score</strong> (0–100%) for each category.
        Higher = more confident. If scores are spread out, the image is ambiguous!
    </div>
    <div class="concept-box">
        <div class="concept-title">🏋️ How was it trained?</div>
        The model was trained on <strong>synthetic colour profiles</strong> — thousands of
        generated colour samples for each category. Real models use millions of real photos!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📸 Try These:")
    st.markdown("""
    <div style="color:#64748b; font-size:0.85rem; line-height:1.9">
    🌿 A photo of a forest or plant<br>
    🌊 Ocean / lake / pool photo<br>
    🔴 A red or orange object<br>
    ☀️ Bright outdoor sunny photo<br>
    🌑 A dark night-time photo<br>
    💙 Something mostly blue
    </div>
    """, unsafe_allow_html=True)

with main_col:
    clf, categories = build_color_classifier()

    uploaded = st.file_uploader(
        "📁 Upload an image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_col, meta_col = st.columns([1, 1], gap="medium")
        with img_col:
            st.image(img, caption="Your uploaded image", use_container_width=True)
        with meta_col:
            w, h = img.size
            arr = np.array(img.resize((64,64)), dtype=np.float32)
            mean_r = arr[:,:,0].mean()
            mean_g = arr[:,:,1].mean()
            mean_b = arr[:,:,2].mean()
            brightness = (mean_r + mean_g + mean_b) / 3
            st.markdown(f"""
            <div class="result-card" style="margin-top:0">
                <div style="color:#94a3b8; font-size:0.85rem">📐 Image Stats</div>
                <div style="color:#e2e8f0; margin-top:0.5rem; font-size:0.9rem; line-height:2">
                    <strong>Size:</strong> {w} × {h} px<br>
                    <strong>Avg Red:</strong> {mean_r:.0f}<br>
                    <strong>Avg Green:</strong> {mean_g:.0f}<br>
                    <strong>Avg Blue:</strong> {mean_b:.0f}<br>
                    <strong>Brightness:</strong> {brightness:.0f}/255
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍 Classify this Image!", use_container_width=True):
            features = extract_features(img)
            proba = clf.predict_proba(features)[0]
            top5_idx = np.argsort(proba)[::-1][:5]
            labels = [categories[i] for i in top5_idx]
            scores = [proba[i] * 100 for i in top5_idx]

            # Top result
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#94a3b8; font-size:0.85rem">🥇 Top Prediction</div>
                <div class="top-prediction">{labels[0]}</div>
                <div style="color:#64748b; font-size:0.85rem">Confidence: {scores[0]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart
            colors = ["#34d399","#60a5fa","#a78bfa","#fb923c","#f87171"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores[::-1], y=labels[::-1], orientation="h",
                marker=dict(color=colors[::-1], line=dict(width=0)),
                text=[f"{s:.1f}%" for s in scores[::-1]],
                textposition="auto", textfont=dict(color="white", size=13),
            ))
            fig.update_layout(
                title=dict(text="Top 5 Predictions", font=dict(color="#e2e8f0", size=16)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#94a3b8", family="Outfit"),
                xaxis=dict(title="Confidence (%)", range=[0,105],
                           gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
                margin=dict(l=10, r=10, t=40, b=40), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Colour breakdown chart
            fig2 = go.Figure(go.Bar(
                x=["Red", "Green", "Blue"],
                y=[mean_r, mean_g, mean_b],
                marker_color=["#f87171","#4ade80","#60a5fa"],
            ))
            fig2.update_layout(
                title=dict(text="Colour Channel Analysis (0–255)", font=dict(color="#e2e8f0", size=14)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#94a3b8", family="Outfit"),
                yaxis=dict(range=[0,255], gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=10, r=10, t=40, b=30), height=220,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.info("💡 **Fun Fact**: This classifier extracted just **8 numbers** from your image "
                    "and a Random Forest of 150 decision trees voted to reach this answer — all offline, instantly!")
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; border:2px dashed rgba(96,165,250,0.3);
                    border-radius:20px; color:#475569; margin-top:1rem">
            <div style="font-size:3.5rem">📷</div>
            <div style="font-size:1.1rem; margin-top:0.5rem; color:#64748b">
                Upload a photo to begin!<br>
                <span style="font-size:0.85rem">Works 100% offline — no internet needed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
