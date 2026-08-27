import io
import math
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from textblob import TextBlob

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ── Global Page Configuration ────────────────────────────────────────────────
st.set_page_config(page_title="AI Lab for Students", page_icon="🧠", layout="wide")

# Auto-load API key from secrets if available
if "api_key" not in st.session_state:
    try:
        st.session_state.api_key = st.secrets["gemini"]["api_key"]
    except Exception:
        pass

# ── Navigation Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("🧠 AI Lab Navigation")
page = st.sidebar.radio(
    "Select a Demo:",
    [
        "🏠 Home",
        "🤖 AI Chatbot",
        "🔍 Image Classifier",
        "📊 Sentiment Analyzer",
        "🌸 Smart Predictor",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); min-height: 100vh; }
    .hero-title {
        font-size: 3.8rem; font-weight: 900;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        line-height: 1.1; margin-bottom: 0.3rem;
    }
    .hero-sub { font-size: 1.25rem; color: #c4b5fd; margin-bottom: 2rem; font-weight: 300; }
    .demo-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px; padding: 1.8rem 1.6rem; height: 100%; transition: all 0.3s ease;
        backdrop-filter: blur(12px);
    }
    .card-emoji { font-size: 2.8rem; margin-bottom: 0.6rem; }
    .card-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem; }
    .card-desc { font-size: 0.92rem; color: #94a3b8; line-height: 1.5; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin-top: 0.8rem; }
    .badge-easy   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
    .badge-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
    .concept-pill {
        display: inline-block; background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.3);
        color: #93c5fd; border-radius: 50px; padding: 4px 14px; font-size: 0.8rem; margin: 3px;
    }
    .info-box {
        background: rgba(96,165,250,0.08); border-left: 4px solid #60a5fa; border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem; margin: 1rem 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.7;
    }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(167,139,250,0.3); }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col_hero, col_img = st.columns([2, 1], gap="large")
    with col_hero:
        st.markdown(
            '<div class="hero-title">🧠 AI Lab for<br>School Students</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-sub">Interactive AI demos for Grade 8, 9 & 10 — learn by doing!</div>',
            unsafe_allow_html=True,
        )
        concepts = [
            "Machine Learning",
            "Neural Networks",
            "NLP",
            "Computer Vision",
            "Classification",
            "Sentiment Analysis",
            "Deep Learning",
        ]
        pills_html = "".join(
            f'<span class="concept-pill">{c}</span>' for c in concepts
        )
        st.markdown(
            f'<div style="margin-bottom:1.5rem">{pills_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <div class="info-box">
        👋 <strong>Welcome, Explorer!</strong> This lab has <strong>4 AI demos</strong> that show you how
        Artificial Intelligence works — no coding skills needed. Pick any demo from the sidebar and start experimenting!
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_img:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
        <div style="text-align:center; font-size:7rem; animation: float 3s ease-in-out infinite;">🤖</div>
        <style>
        @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-18px); } }
        </style>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div class="section-header">🚀 Choose a Demo</div>',
        unsafe_allow_html=True,
    )

    cards = [
        {
            "emoji": "🤖",
            "title": "AI Chatbot",
            "desc": "Chat with an AI that understands your questions! Learn how Large Language Models work.",
            "concepts": ["LLM", "NLP", "Prompting"],
            "badge": "easy",
            "grade": "Grades 8–10",
        },
        {
            "emoji": "🔍",
            "title": "Image Classifier",
            "desc": "Upload any photo and watch the AI identify objects using Computer Vision.",
            "concepts": ["CNN", "Deep Learning", "Confidence"],
            "badge": "medium",
            "grade": "Grades 9–10",
        },
        {
            "emoji": "📊",
            "title": "Sentiment Analyzer",
            "desc": "Type any sentence and the AI will figure out if it's positive, negative, or neutral.",
            "concepts": ["NLP", "Text Mining", "Classification"],
            "badge": "easy",
            "grade": "Grades 8–9",
        },
        {
            "emoji": "🌸",
            "title": "Smart Predictor",
            "desc": "Move sliders to describe a flower and the ML model instantly predicts its species!",
            "concepts": ["Supervised ML", "Decision Tree", "Features"],
            "badge": "medium",
            "grade": "Grades 9–10",
        },
    ]

    cols = st.columns(4, gap="medium")
    for col, card in zip(cols, cards):
        badge_class = f"badge-{card['badge']}"
        pills = "".join(
            f'<span class="concept-pill" style="font-size:0.7rem;padding:2px 8px">{c}</span>'
            for c in card["concepts"]
        )
        with col:
            st.markdown(
                f"""
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
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<div class="section-header">📖 What is Artificial Intelligence?</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            """
        <div class="demo-card">
            <div class="card-emoji">🎓</div>
            <div class="card-title">AI Learns from Data</div>
            <div class="card-desc">Just like you learn from examples in class, AI learns patterns from millions of examples called <em>training data</em>.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
        <div class="demo-card">
            <div class="card-emoji">🔮</div>
            <div class="card-title">AI Makes Predictions</div>
            <div class="card-desc">After learning, the AI can <em>predict</em> answers for new inputs — like recognising a photo it has never seen before.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
        <div class="demo-card">
            <div class="card-emoji">♾️</div>
            <div class="card-title">AI Keeps Improving</div>
            <div class="card-desc">With more data and feedback, AI models <em>improve over time</em> — similar to how you get better at a skill with practice.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 AI Chatbot":
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .page-title { font-size: 2.6rem; font-weight: 900; background: linear-gradient(90deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
    .concept-box { background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.25); border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: #c4b5fd; font-size: 0.9rem; line-height: 1.6; }
    .concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #a78bfa; }
    .user-bubble { background: linear-gradient(135deg, #7c3aed, #4f46e5); border-radius: 18px 18px 4px 18px; padding: 0.7rem 1.1rem; margin: 0.4rem 0 0.4rem auto; max-width: 75%; color: #fff; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(124,58,237,0.3); }
    .bot-bubble { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 18px 18px 18px 4px; padding: 0.7rem 1.1rem; margin: 0.4rem auto 0.4rem 0; max-width: 78%; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .chat-label { font-size: 0.72rem; color: #64748b; margin: 2px 4px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">🤖 AI Chatbot</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">Chat with an AI assistant and learn how Large Language Models work!</div>',
        unsafe_allow_html=True,
    )

    main_col, sidebar_col = st.columns([2, 1], gap="large")

    with sidebar_col:
        st.markdown(
            """
        <div class="concept-box">
            <div class="concept-title">💡 What is an LLM?</div>
            A <strong>Large Language Model (LLM)</strong> is an AI trained on billions of text examples.
            It learns patterns in language to predict what word comes next — that's how it generates answers!
        </div>
        <div class="concept-box">
            <div class="concept-title">🔤 What is a Prompt?</div>
            A <strong>prompt</strong> is the message you send to the AI. The better your prompt, the better
            the answer! Try being specific about what you want.
        </div>
        <div class="concept-box">
            <div class="concept-title">🌡️ What is a Token?</div>
            AI doesn't read word-by-word. It breaks text into small pieces called <strong>tokens</strong>
            (roughly 1 token ≈ ¾ of a word). "Hello world" = 2 tokens.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 💬 Starter Questions")
        starters = [
            "Explain gravity in simple terms",
            "How do plants make food?",
            "What is the water cycle?",
            "Tell me a fun science fact",
            "How does the internet work?",
        ]
        for q in starters:
            if st.button(q, key=f"starter_{q}", use_container_width=True):
                st.session_state.starter_q = q

    with main_col:
        key_loaded = "api_key" in st.session_state
        with st.expander("🔑 Gemini API Key", expanded=not key_loaded):
            if key_loaded and st.session_state.api_key != "DEMO":
                st.success("✅ API key loaded automatically from secrets!")
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste your API key here…",
                help="Get a free key at https://aistudio.google.com/",
                label_visibility="collapsed",
            )
            st.caption("🔒 Your key is stored only in this session.")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if (
                    st.button("✅ Connect", use_container_width=True)
                    and api_key_input
                ):
                    st.session_state.api_key = api_key_input
                    st.session_state.messages = []
                    st.success("Connected! Start chatting below 🎉")
                    st.rerun()
            with col_b:
                if st.button(
                    "🎭 Demo Mode (No Key)", use_container_width=True
                ):
                    st.session_state.api_key = "DEMO"
                    st.session_state.messages = []
                    st.info("Demo mode: canned responses only.")
                    st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        SYSTEM = (
            "You are 'Aria', a friendly AI tutor for school students in grades 8–10. "
            "You explain concepts clearly using simple language, real-world analogies, "
            "and occasional emojis. Keep answers concise (3–5 sentences) unless the student asks for more. "
            "Always encourage curiosity and end with a follow-up question when appropriate."
        )

        DEMO_RESPONSES = {
            "default": "🤖 I'm in demo mode! In the real app (with an API key), I can answer any question. "
            "Try asking about science, math, history — anything!",
        }

        def get_ai_response(prompt: str) -> str:
            if st.session_state.get("api_key") == "DEMO":
                return DEMO_RESPONSES["default"]
            if not genai:
                return "⚠️ google-generativeai package is not installed."
            try:
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    system_instruction=SYSTEM,
                )
                history = []
                for msg in st.session_state.messages[:-1]:
                    history.append(
                        {"role": msg["role"], "parts": [msg["content"]]}
                    )
                chat = model.start_chat(history=history)
                response = chat.send_message(prompt)
                return response.text
            except Exception as e:
                return f"⚠️ Error: {e}\n\nPlease check your API key and try again."

        if "starter_q" in st.session_state:
            starter = st.session_state.pop("starter_q")
            st.session_state.messages.append(
                {"role": "user", "content": starter}
            )
            if "api_key" in st.session_state:
                reply = get_ai_response(starter)
                st.session_state.messages.append(
                    {"role": "model", "content": reply}
                )

        if "api_key" in st.session_state:
            chat_container = st.container(height=480)
            with chat_container:
                if not st.session_state.messages:
                    st.markdown(
                        """
                    <div style="text-align:center; padding:3rem; color:#475569;">
                        <div style="font-size:3rem">👋</div>
                        <div style="font-size:1.1rem; margin-top:0.5rem">Hi! I'm <strong style="color:#a78bfa">Aria</strong>,
                        your AI tutor.<br>Ask me anything about science, math, or any school subject!</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            st.markdown(
                                '<div class="chat-label" style="text-align:right">You</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="user-bubble">{msg["content"]}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="chat-label">🤖 Aria</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="bot-bubble">{msg["content"]}</div>',
                                unsafe_allow_html=True,
                            )

            inp_col, btn_col = st.columns([5, 1], gap="small")
            with inp_col:
                user_input = st.text_input(
                    "Your message",
                    key="chat_input",
                    placeholder="Ask Aria anything…",
                    label_visibility="collapsed",
                )
            with btn_col:
                send = st.button("Send ➤", use_container_width=True)

            if send and user_input.strip():
                st.session_state.messages.append(
                    {"role": "user", "content": user_input.strip()}
                )
                with st.spinner("Aria is thinking…"):
                    reply = get_ai_response(user_input.strip())
                st.session_state.messages.append(
                    {"role": "model", "content": reply}
                )
                st.rerun()

            if st.session_state.messages:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.info(
                "👆 Please enter your Gemini API key or use Demo Mode to start chatting!"
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: IMAGE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Image Classifier":
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
    .page-title { font-size: 2.6rem; font-weight: 900; background: linear-gradient(90deg, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
    .concept-box { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.2); border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: #a7f3d0; font-size: 0.9rem; line-height: 1.6; }
    .concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #34d399; }
    .result-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(52,211,153,0.3); border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem; }
    .top-prediction { font-size: 2rem; font-weight: 900; background: linear-gradient(90deg, #60a5fa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def build_color_classifier():
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

        def make_samples(mean_rgb, n=120, noise=30):
            samples = []
            for _ in range(n):
                r = np.clip(mean_rgb[0] + random.gauss(0, noise), 0, 255)
                g = np.clip(mean_rgb[1] + random.gauss(0, noise), 0, 255)
                b = np.clip(mean_rgb[2] + random.gauss(0, noise), 0, 255)
                bright = (r + g + b) / 3
                total = r + g + b + 1e-6
                samples.append(
                    [
                        r,
                        g,
                        b,
                        bright,
                        abs(r - g) + abs(g - b),
                        r / total,
                        g / total,
                        b / total,
                    ]
                )
            return samples

        profiles = [
            (60, 130, 60),
            (50, 100, 180),
            (220, 90, 30),
            (30, 30, 40),
            (230, 220, 150),
            (200, 50, 50),
            (230, 210, 50),
            (50, 100, 220),
            (230, 230, 230),
            (30, 30, 30),
            (140, 90, 60),
            (200, 100, 180),
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
        img_small = img.resize((64, 64)).convert("RGB")
        arr = np.array(img_small, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
        brightness = (mean_r + mean_g + mean_b) / 3
        contrast = arr.std()
        total = mean_r + mean_g + mean_b + 1e-6
        return np.array(
            [
                [
                    mean_r,
                    mean_g,
                    mean_b,
                    brightness,
                    contrast,
                    mean_r / total,
                    mean_g / total,
                    mean_b / total,
                ]
            ]
        )

    st.markdown(
        '<div class="page-title">🔍 Image Classifier</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">Upload any photo and watch the AI analyse its colours and patterns to classify it!</div>',
        unsafe_allow_html=True,
    )

    main_col, sidebar_col = st.columns([3, 2], gap="large")

    with sidebar_col:
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )

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
                st.image(
                    img, caption="Your uploaded image", use_container_width=True
                )
            with meta_col:
                w, h = img.size
                arr = np.array(img.resize((64, 64)), dtype=np.float32)
                mean_r = arr[:, :, 0].mean()
                mean_g = arr[:, :, 1].mean()
                mean_b = arr[:, :, 2].mean()
                brightness = (mean_r + mean_g + mean_b) / 3
                st.markdown(
                    f"""
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
                """,
                    unsafe_allow_html=True,
                )

            if st.button("🔍 Classify this Image!", use_container_width=True):
                features = extract_features(img)
                proba = clf.predict_proba(features)[0]
                top5_idx = np.argsort(proba)[::-1][:5]
                labels = [categories[i] for i in top5_idx]
                scores = [proba[i] * 100 for i in top5_idx]

                st.markdown(
                    f"""
                <div class="result-card">
                    <div style="color:#94a3b8; font-size:0.85rem">🥇 Top Prediction</div>
                    <div class="top-prediction">{labels[0]}</div>
                    <div style="color:#64748b; font-size:0.85rem">Confidence: {scores[0]:.1f}%</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                colors = [
                    "#34d399",
                    "#60a5fa",
                    "#a78bfa",
                    "#fb923c",
                    "#f87171",
                ]
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=scores[::-1],
                        y=labels[::-1],
                        orientation="h",
                        marker=dict(color=colors[::-1], line=dict(width=0)),
                        text=[f"{s:.1f}%" for s in scores[::-1]],
                        textposition="auto",
                        textfont=dict(color="white", size=13),
                    )
                )
                fig.update_layout(
                    title=dict(
                        text="Top 5 Predictions",
                        font=dict(color="#e2e8f0", size=16),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="#94a3b8", family="Outfit"),
                    xaxis=dict(
                        title="Confidence (%)",
                        range=[0, 105],
                        gridcolor="rgba(255,255,255,0.06)",
                        tickfont=dict(color="#94a3b8"),
                    ),
                    yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure(
                    go.Bar(
                        x=["Red", "Green", "Blue"],
                        y=[mean_r, mean_g, mean_b],
                        marker_color=["#f87171", "#4ade80", "#60a5fa"],
                    )
                )
                fig2.update_layout(
                    title=dict(
                        text="Colour Channel Analysis (0–255)",
                        font=dict(color="#e2e8f0", size=14),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="#94a3b8", family="Outfit"),
                    yaxis=dict(
                        range=[0, 255], gridcolor="rgba(255,255,255,0.06)"
                    ),
                    margin=dict(l=10, r=10, t=40, b=30),
                    height=220,
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.info(
                    "💡 **Fun Fact**: This classifier extracted just **8 numbers** from your image "
                    "and a Random Forest of 150 decision trees voted to reach this answer — all offline, instantly!"
                )
        else:
            st.markdown(
                """
            <div style="text-align:center; padding:3rem; border:2px dashed rgba(96,165,250,0.3);
                        border-radius:20px; color:#475569; margin-top:1rem">
                <div style="font-size:3.5rem">📷</div>
                <div style="font-size:1.1rem; margin-top:0.5rem; color:#64748b">
                    Upload a photo to begin!<br>
                    <span style="font-size:0.85rem">Works 100% offline — no internet needed</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: SENTIMENT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Sentiment Analyzer":
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #1a0533 0%, #2d1b69 50%, #1e1b4b 100%); }
    .page-title { font-size: 2.6rem; font-weight: 900; background: linear-gradient(90deg, #f472b6, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
    .concept-box { background: rgba(244,114,182,0.07); border: 1px solid rgba(244,114,182,0.2); border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: #fbcfe8; font-size: 0.9rem; line-height: 1.6; }
    .concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #f472b6; }
    .sentiment-display { text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }
    .sentiment-emoji  { font-size: 4rem; }
    .sentiment-label  { font-size: 2rem; font-weight: 900; margin: 0.3rem 0; }
    .sentiment-pos    { color: #34d399; }
    .sentiment-neg    { color: #f87171; }
    .sentiment-neu    { color: #fbbf24; }
    .metric-box { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1rem; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: 800; color: #f1f5f9; }
    .metric-lbl { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
    .word-highlight-pos { background: rgba(52,211,153,0.2); border-radius:4px; padding:1px 3px; color:#34d399; }
    .word-highlight-neg { background: rgba(248,113,113,0.2); border-radius:4px; padding:1px 3px; color:#f87171; }
    .word-highlight-neu { color: #cbd5e1; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    POS_WORDS = {
        "great",
        "good",
        "amazing",
        "excellent",
        "happy",
        "love",
        "wonderful",
        "fantastic",
        "best",
        "awesome",
        "beautiful",
        "brilliant",
        "nice",
        "enjoy",
        "excited",
        "grateful",
        "positive",
        "perfect",
        "incredible",
        "joy",
        "fun",
        "super",
        "cool",
        "like",
        "thank",
        "pleased",
        "delighted",
        "proud",
        "kind",
    }
    NEG_WORDS = {
        "bad",
        "terrible",
        "horrible",
        "awful",
        "hate",
        "worst",
        "sad",
        "angry",
        "upset",
        "disgusting",
        "poor",
        "negative",
        "ugly",
        "boring",
        "disappointing",
        "dull",
        "wrong",
        "fail",
        "annoying",
        "frustrating",
        "useless",
        "stupid",
        "painful",
        "fear",
        "cry",
    }

    def highlight_words(text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in POS_WORDS:
                result.append(f'<span class="word-highlight-pos">{w}</span>')
            elif clean in NEG_WORDS:
                result.append(f'<span class="word-highlight-neg">{w}</span>')
            else:
                result.append(f'<span class="word-highlight-neu">{w}</span>')
        return " ".join(result)

    def analyze(text: str):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0.1:
            label, emoji, css = "Positive", "😄", "sentiment-pos"
        elif polarity < -0.1:
            label, emoji, css = "Negative", "😞", "sentiment-neg"
        else:
            label, emoji, css = "Neutral", "😐", "sentiment-neu"
        return polarity, subjectivity, label, emoji, css

    st.markdown(
        '<div class="page-title">📊 Sentiment Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">Type any sentence and the AI will detect if it\'s positive, negative, or neutral!</div>',
        unsafe_allow_html=True,
    )

    main_col, sidebar_col = st.columns([3, 2], gap="large")

    with sidebar_col:
        st.markdown(
            """
        <div class="concept-box">
            <div class="concept-title">❤️ What is Sentiment?</div>
            <strong>Sentiment</strong> is the emotional tone of a piece of text — is the writer happy,
            sad, or neutral? AI can detect this by analysing word choice and patterns.
        </div>
        <div class="concept-box">
            <div class="concept-title">🎭 Polarity vs Subjectivity</div>
            <strong>Polarity</strong>: How positive (+1) or negative (−1) the text is.<br><br>
            <strong>Subjectivity</strong>: Is it a fact (0) or an opinion (1)?<br>
            <em>"The sky is blue"</em> = objective.<br>
            <em>"The sky is beautiful"</em> = subjective.
        </div>
        <div class="concept-box">
            <div class="concept-title">🔤 How does it work?</div>
            TextBlob uses a <strong>lexicon</strong> — a dictionary of words with pre-assigned
            sentiment scores. It sums these scores across all words to compute the overall sentiment.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 💬 Try These Sentences")
        examples = [
            "I absolutely love learning new things every day!",
            "This movie was a complete waste of time.",
            "The Earth orbits the Sun once every 365 days.",
            "The food was okay, nothing special.",
            "I'm so excited about the science fair tomorrow!",
        ]
        for ex in examples:
            if st.button(
                f'"{ex[:40]}…"' if len(ex) > 40 else f'"{ex}"',
                key=f"ex_{ex[:20]}",
                use_container_width=True,
            ):
                st.session_state.sentiment_example = ex

    with main_col:
        default_val = st.session_state.pop("sentiment_example", "")
        user_text = st.text_area(
            "✍️ Enter your text here:",
            value=default_val,
            placeholder="Type a sentence, paragraph, or even a tweet…",
            height=130,
            key="sentiment_input",
        )
        analyze_btn = st.button(
            "🔍 Analyze Sentiment!", use_container_width=True
        )

        if analyze_btn and user_text.strip():
            polarity, subjectivity, label, emoji, css = analyze(
                user_text.strip()
            )

            st.markdown(
                f"""
            <div class="sentiment-display">
                <div class="sentiment-emoji">{emoji}</div>
                <div class="sentiment-label {css}">{label}</div>
                <div style="color:#64748b; font-size:0.9rem">Sentiment detected</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3, gap="medium")
            with m1:
                pol_color = (
                    "#34d399"
                    if polarity > 0.1
                    else "#f87171"
                    if polarity < -0.1
                    else "#fbbf24"
                )
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-val" style="color:{pol_color}">{polarity:+.2f}</div>
                    <div class="metric-lbl">Polarity (−1 to +1)</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-val">{subjectivity:.2f}</div>
                    <div class="metric-lbl">Subjectivity (0=fact, 1=opinion)</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                word_count = len(user_text.split())
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-val">{word_count}</div>
                    <div class="metric-lbl">Words Analyzed</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=polarity,
                    number=dict(
                        suffix="", font=dict(color="#e2e8f0", size=28)
                    ),
                    gauge=dict(
                        axis=dict(
                            range=[-1, 1],
                            tickcolor="#64748b",
                            tickfont=dict(color="#64748b"),
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            ticktext=[
                                "Very\nNegative",
                                "Negative",
                                "Neutral",
                                "Positive",
                                "Very\nPositive",
                            ],
                        ),
                        bar=dict(color=pol_color, thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        steps=[
                            dict(
                                range=[-1, -0.1],
                                color="rgba(248,113,113,0.15)",
                            ),
                            dict(
                                range=[-0.1, 0.1],
                                color="rgba(251,191,36,0.15)",
                            ),
                            dict(range=[0.1, 1], color="rgba(52,211,153,0.15)"),
                        ],
                        threshold=dict(
                            line=dict(color=pol_color, width=3),
                            thickness=0.75,
                            value=polarity,
                        ),
                    ),
                    title=dict(
                        text="Sentiment Polarity Gauge",
                        font=dict(color="#94a3b8", size=14),
                    ),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color="#94a3b8"),
                height=220,
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🔬 Word Analysis")
            st.markdown(
                """
            <div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:1rem 1.2rem;
                        font-size:1rem; line-height:2; border:1px solid rgba(255,255,255,0.08)">
            """
                + highlight_words(user_text.strip())
                + """
            </div>
            <div style="margin-top:0.5rem; font-size:0.8rem; color:#64748b">
                <span style="background:rgba(52,211,153,0.2); border-radius:4px; padding:1px 6px; color:#34d399">Green</span> = positive word &nbsp;&nbsp;
                <span style="background:rgba(248,113,113,0.2); border-radius:4px; padding:1px 6px; color:#f87171">Red</span> = negative word
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif analyze_btn:
            st.warning("⚠️ Please enter some text first!")
        else:
            st.markdown(
                """
            <div style="text-align:center; padding:3rem; border:2px dashed rgba(244,114,182,0.3);
                        border-radius:20px; color:#475569; margin-top:1rem">
                <div style="font-size:3.5rem">✍️</div>
                <div style="font-size:1.1rem; margin-top:0.5rem; color:#64748b">
                    Type something and click Analyze!<br>
                    <span style="font-size:0.85rem">Try a happy sentence, a sad one, or a fact</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: SMART PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌸 Smart Predictor":
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0c1a0e 0%, #14532d 30%, #064e3b 70%, #0c1a0e 100%); }
    .page-title { font-size: 2.6rem; font-weight: 900; background: linear-gradient(90deg, #4ade80, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }
    .concept-box { background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.2); border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: #bbf7d0; font-size: 0.9rem; line-height: 1.6; }
    .concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #4ade80; }
    .prediction-card { background: rgba(255,255,255,0.05); border-radius: 24px; padding: 2rem; text-align: center; border: 2px solid; margin-bottom: 1rem; }
    .pred-species  { font-size: 2.2rem; font-weight: 900; margin: 0.5rem 0; }
    .pred-conf     { color: #94a3b8; font-size: 0.9rem; }
    .metric-box { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1rem; text-align: center; }
    .metric-val { font-size: 1.6rem; font-weight: 800; }
    .metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .slider-label { color: #a7f3d0; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

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
    SPECIES_BORDER = [
        "rgba(74,222,128,0.5)",
        "rgba(167,139,250,0.5)",
        "rgba(244,114,182,0.5)",
    ]

    st.markdown(
        '<div class="page-title">🌸 Smart Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">Adjust the flower measurements and let the AI predict the Iris species!</div>',
        unsafe_allow_html=True,
    )

    main_col, sidebar_col = st.columns([3, 2], gap="large")

    with sidebar_col:
        st.markdown(
            """
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
        """.format(
                accuracy * 100
            ),
            unsafe_allow_html=True,
        )

        with st.expander("📊 View Raw Dataset (first 10 rows)", expanded=False):
            df = pd.DataFrame(
                iris_data.data,
                columns=[
                    "Sepal Length",
                    "Sepal Width",
                    "Petal Length",
                    "Petal Width",
                ],
            )
            df["Species"] = [SPECIES[t] for t in iris_data.target]
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    with main_col:
        st.markdown("#### 🎛️ Adjust Flower Measurements")
        st.markdown(
            """
        <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem">
        Move the sliders to set the measurements (in centimetres) and watch the AI predict in real-time!
        </div>
        """,
            unsafe_allow_html=True,
        )

        sl_col1, sl_col2 = st.columns(2, gap="medium")
        with sl_col1:
            st.markdown(
                '<div class="slider-label">📏 Sepal Length (cm)</div>',
                unsafe_allow_html=True,
            )
            sepal_length = st.slider(
                "Sepal Length",
                4.0,
                8.0,
                5.8,
                0.1,
                key="sl",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="slider-label">📏 Sepal Width (cm)</div>',
                unsafe_allow_html=True,
            )
            sepal_width = st.slider(
                "Sepal Width",
                2.0,
                4.5,
                3.0,
                0.1,
                key="sw",
                label_visibility="collapsed",
            )

        with sl_col2:
            st.markdown(
                '<div class="slider-label">🌿 Petal Length (cm)</div>',
                unsafe_allow_html=True,
            )
            petal_length = st.slider(
                "Petal Length",
                1.0,
                7.0,
                4.0,
                0.1,
                key="pl",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="slider-label">🌿 Petal Width (cm)</div>',
                unsafe_allow_html=True,
            )
            petal_width = st.slider(
                "Petal Width",
                0.1,
                2.5,
                1.3,
                0.1,
                key="pw",
                label_visibility="collapsed",
            )

        features = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]]
        )
        pred_class = rf_model.predict(features)[0]
        pred_proba = rf_model.predict_proba(features)[0]
        confidence = pred_proba[pred_class] * 100

        species_name = SPECIES[pred_class]
        species_emoji = SPECIES_EMOJI[pred_class]
        pred_color = SPECIES_COLORS[pred_class]
        pred_border = SPECIES_BORDER[pred_class]

        st.markdown(
            f"""
        <div class="prediction-card" style="border-color: {pred_border}; background: rgba(0,0,0,0.2)">
            <div style="font-size:4rem">{species_emoji}</div>
            <div class="pred-species" style="color: {pred_color}">{species_name}</div>
            <div class="pred-conf">Confidence: {confidence:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        prob_col1, prob_col2, prob_col3 = st.columns(3, gap="small")
        for col, i in zip([prob_col1, prob_col2, prob_col3], range(3)):
            prob_pct = pred_proba[i] * 100
            is_top = i == pred_class
            with col:
                st.markdown(
                    f"""
                <div class="metric-box" style="border-color:{'rgba(255,255,255,0.3)' if is_top else 'rgba(255,255,255,0.08)'}">
                    <div style="font-size:1.5rem">{SPECIES_EMOJI[i]}</div>
                    <div class="metric-val" style="color:{SPECIES_COLORS[i]}">{prob_pct:.0f}%</div>
                    <div class="metric-lbl">{SPECIES[i].split()[-1]}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(
            ["📊 Feature Importance", "🌐 Dataset Scatter", "🌲 Decision Tree"]
        )

        with tab1:
            importances = rf_model.feature_importances_
            feature_names = [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
            ]
            sorted_idx = np.argsort(importances)
            fig_imp = go.Figure(
                go.Bar(
                    x=importances[sorted_idx],
                    y=[feature_names[i] for i in sorted_idx],
                    orientation="h",
                    marker=dict(
                        color=["#a78bfa", "#60a5fa", "#4ade80", "#34d399"][
                            : len(sorted_idx)
                        ],
                        line=dict(width=0),
                    ),
                    text=[f"{v*100:.1f}%" for v in importances[sorted_idx]],
                    textposition="auto",
                    textfont=dict(color="white", size=13),
                )
            )
            fig_imp.update_layout(
                title=dict(
                    text="Which features matter most to the AI?",
                    font=dict(color="#e2e8f0", size=14),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#94a3b8", family="Outfit"),
                xaxis=dict(
                    title="Importance",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="#94a3b8"),
                ),
                yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
                margin=dict(l=10, r=10, t=40, b=30),
                height=250,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info(
                "💡 **Petal measurements** matter much more than sepal measurements for this classification!"
            )

        with tab2:
            df_plot = pd.DataFrame(
                iris_data.data,
                columns=[
                    "Sepal Length",
                    "Sepal Width",
                    "Petal Length",
                    "Petal Width",
                ],
            )
            df_plot["Species"] = [SPECIES[t] for t in iris_data.target]

            fig_scatter = px.scatter(
                df_plot,
                x="Petal Length",
                y="Petal Width",
                color="Species",
                color_discrete_map={
                    s: c for s, c in zip(SPECIES, SPECIES_COLORS)
                },
                title="Iris Dataset: Petal Length vs Petal Width",
                labels={
                    "Petal Length": "Petal Length (cm)",
                    "Petal Width": "Petal Width (cm)",
                },
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[petal_length],
                    y=[petal_width],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color=pred_color,
                        line=dict(color="white", width=2),
                    ),
                    name="Your Flower ⭐",
                )
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#94a3b8", family="Outfit"),
                legend=dict(
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1,
                ),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                height=320,
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                "⭐ The star shows where YOUR flower measurements fall in the dataset!"
            )

        with tab3:
            st.markdown("#### 🌲 Simplified Decision Tree (depth 3)")
            tree_text = export_text(
                dt_model,
                feature_names=[
                    "Sepal Len",
                    "Sepal Wid",
                    "Petal Len",
                    "Petal Wid",
                ],
            )
            st.code(tree_text, language="")
            st.info(
                "💡 This tree shows the if-else rules the AI learned. The Random Forest uses 100 such trees!"
            )
