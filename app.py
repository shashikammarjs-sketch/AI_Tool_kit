import streamlit as st

st.set_page_config(
    page_title="AI Lab for Students",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hero section */
.hero-title {
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.25rem;
    color: #c4b5fd;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Demo cards */
.demo-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    height: 100%;
    transition: all 0.3s ease;
    backdrop-filter: blur(12px);
    cursor: pointer;
}
.demo-card:hover {
    background: rgba(167,139,250,0.12);
    border-color: #a78bfa;
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(167,139,250,0.2);
}
.card-emoji {
    font-size: 2.8rem;
    margin-bottom: 0.6rem;
}
.card-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.card-desc {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.5;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 0.8rem;
}
.badge-easy   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
.badge-medium { background: rgba(251,191, 36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
.badge-hard   { background: rgba(239, 68, 68,0.15); color: #f87171; border: 1px solid #f87171; }

/* Concept pill */
.concept-pill {
    display: inline-block;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.3);
    color: #93c5fd;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.8rem;
    margin: 3px;
}

/* Info box */
.info-box {
    background: rgba(96,165,250,0.08);
    border-left: 4px solid #60a5fa;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Section header */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(167,139,250,0.3);
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
col_hero, col_img = st.columns([2, 1], gap="large")

with col_hero:
    st.markdown('<div class="hero-title">🧠 AI Lab for<br>School Students</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Interactive AI demos for Grade 8, 9 & 10 — learn by doing!</div>', unsafe_allow_html=True)

    concepts = ["Machine Learning", "Neural Networks", "NLP", "Computer Vision",
                "Classification", "Sentiment Analysis", "Deep Learning"]
    pills_html = "".join(f'<span class="concept-pill">{c}</span>' for c in concepts)
    st.markdown(f'<div style="margin-bottom:1.5rem">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    👋 <strong>Welcome, Explorer!</strong> This lab has <strong>4 AI demos</strong> that show you how
    Artificial Intelligence works — no coding skills needed. Pick any demo from the sidebar or the
    cards below and start experimenting!
    </div>
    """, unsafe_allow_html=True)

with col_img:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; font-size:7rem; animation: float 3s ease-in-out infinite;">
    🤖
    </div>
    <style>
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50%      { transform: translateY(-18px); }
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Demo Cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Choose a Demo</div>', unsafe_allow_html=True)

cards = [
    {
        "emoji": "🤖",
        "title": "AI Chatbot",
        "desc": "Chat with an AI that understands your questions! Learn how Large Language Models work.",
        "concepts": ["LLM", "NLP", "Prompting"],
        "badge": "easy",
        "grade": "Grades 8–10",
        "page": "pages/1_🤖_AI_Chatbot.py",
    },
    {
        "emoji": "🔍",
        "title": "Image Classifier",
        "desc": "Upload any photo and watch the AI identify objects using Computer Vision.",
        "concepts": ["CNN", "Deep Learning", "Confidence"],
        "badge": "medium",
        "grade": "Grades 9–10",
        "page": "pages/2_🔍_Image_Classifier.py",
    },
    {
        "emoji": "📊",
        "title": "Sentiment Analyzer",
        "desc": "Type any sentence and the AI will figure out if it's positive, negative, or neutral.",
        "concepts": ["NLP", "Text Mining", "Classification"],
        "badge": "easy",
        "grade": "Grades 8–9",
        "page": "pages/3_📊_Sentiment_Analyzer.py",
    },
    {
        "emoji": "🌸",
        "title": "Smart Predictor",
        "desc": "Move sliders to describe a flower and the ML model instantly predicts its species!",
        "concepts": ["Supervised ML", "Decision Tree", "Features"],
        "badge": "medium",
        "grade": "Grades 9–10",
        "page": "pages/4_🌸_Smart_Predictor.py",
    },
]

cols = st.columns(4, gap="medium")
for col, card in zip(cols, cards):
    badge_class = f"badge-{card['badge']}"
    pills = "".join(f'<span class="concept-pill" style="font-size:0.7rem;padding:2px 8px">{c}</span>' for c in card["concepts"])
    with col:
        st.markdown(f"""
        <div class="demo-card">
            <div class="card-emoji">{card['emoji']}</div>
            <div class="card-title">{card['title']}</div>
            <div class="card-desc">{card['desc']}</div>
            <div style="margin-top:0.8rem">{pills}</div>
            <div>
                <span class="badge {badge_class}">{card['badge'].upper()}</span>
                <span style="color:#64748b; font-size:0.75rem; margin-left:8px">{card['grade']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"Open {card['emoji']} {card['title']}", key=f"btn_{card['title']}", use_container_width=True):
            st.switch_page(card["page"])

st.markdown("---")

# ── What is AI section ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📖 What is Artificial Intelligence?</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("""
    <div class="demo-card">
        <div class="card-emoji">🎓</div>
        <div class="card-title">AI Learns from Data</div>
        <div class="card-desc">Just like you learn from examples in class, AI learns patterns from
        millions of examples called <em>training data</em>.</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="demo-card">
        <div class="card-emoji">🔮</div>
        <div class="card-title">AI Makes Predictions</div>
        <div class="card-desc">After learning, the AI can <em>predict</em> answers for new inputs —
        like recognising a photo it has never seen before.</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="demo-card">
        <div class="card-emoji">♾️</div>
        <div class="card-title">AI Keeps Improving</div>
        <div class="card-desc">With more data and feedback, AI models <em>improve over time</em> —
        similar to how you get better at a skill with practice.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.85rem">
    Made with ❤️ for curious minds · AI Lab for School Students (Grades 8–10)
</div>
""", unsafe_allow_html=True)
