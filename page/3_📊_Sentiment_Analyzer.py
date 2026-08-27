import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
import re
from collections import Counter

st.set_page_config(page_title="Sentiment Analyzer | AI Lab", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #1a0533 0%, #2d1b69 50%, #1e1b4b 100%); }

.page-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #f472b6, #fb923c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }

.concept-box {
    background: rgba(244,114,182,0.07);
    border: 1px solid rgba(244,114,182,0.2);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    color: #fbcfe8; font-size: 0.9rem; line-height: 1.6;
}
.concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #f472b6; }

.sentiment-display {
    text-align: center; padding: 2rem;
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    margin: 1rem 0;
}
.sentiment-emoji  { font-size: 4rem; }
.sentiment-label  { font-size: 2rem; font-weight: 900; margin: 0.3rem 0; }
.sentiment-pos    { color: #34d399; }
.sentiment-neg    { color: #f87171; }
.sentiment-neu    { color: #fbbf24; }

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 800; color: #f1f5f9; }
.metric-lbl { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

.word-highlight-pos { background: rgba(52,211,153,0.2); border-radius:4px; padding:1px 3px; color:#34d399; }
.word-highlight-neg { background: rgba(248,113,113,0.2); border-radius:4px; padding:1px 3px; color:#f87171; }
.word-highlight-neu { color: #cbd5e1; }

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(244,114,182,0.4) !important;
    color: #f1f5f9 !important; border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #be185d, #9333ea) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Positive / Negative word lists for highlighting ───────────────────────────
POS_WORDS = {"great","good","amazing","excellent","happy","love","wonderful","fantastic","best","awesome",
             "beautiful","brilliant","nice","enjoy","excited","grateful","positive","perfect","incredible",
             "joy","fun","super","cool","like","thank","pleased","delighted","proud","kind"}
NEG_WORDS = {"bad","terrible","horrible","awful","hate","worst","sad","angry","upset","disgusting",
             "poor","negative","ugly","boring","disappointing","dull","wrong","fail","horrible",
             "annoying","frustrating","useless","stupid","horrible","awful","painful","fear","cry"}

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
    polarity = blob.sentiment.polarity      # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    if polarity > 0.1:
        label, emoji, css = "Positive", "😄", "sentiment-pos"
    elif polarity < -0.1:
        label, emoji, css = "Negative", "😞", "sentiment-neg"
    else:
        label, emoji, css = "Neutral", "😐", "sentiment-neu"

    return polarity, subjectivity, label, emoji, css

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Type any sentence and the AI will detect if it\'s positive, negative, or neutral!</div>', unsafe_allow_html=True)

main_col, sidebar_col = st.columns([3, 2], gap="large")

with sidebar_col:
    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("#### 💬 Try These Sentences")
    examples = [
        "I absolutely love learning new things every day!",
        "This movie was a complete waste of time.",
        "The Earth orbits the Sun once every 365 days.",
        "The food was okay, nothing special.",
        "I'm so excited about the science fair tomorrow!",
    ]
    for ex in examples:
        if st.button(f'"{ex[:40]}…"' if len(ex) > 40 else f'"{ex}"', key=f"ex_{ex[:20]}", use_container_width=True):
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

    analyze_btn = st.button("🔍 Analyze Sentiment!", use_container_width=True)

    if analyze_btn and user_text.strip():
        polarity, subjectivity, label, emoji, css = analyze(user_text.strip())

        # Main sentiment display
        st.markdown(f"""
        <div class="sentiment-display">
            <div class="sentiment-emoji">{emoji}</div>
            <div class="sentiment-label {css}">{label}</div>
            <div style="color:#64748b; font-size:0.9rem">Sentiment detected</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3, gap="medium")
        with m1:
            pol_color = "#34d399" if polarity > 0.1 else "#f87171" if polarity < -0.1 else "#fbbf24"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="color:{pol_color}">{polarity:+.2f}</div>
                <div class="metric-lbl">Polarity (−1 to +1)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{subjectivity:.2f}</div>
                <div class="metric-lbl">Subjectivity (0=fact, 1=opinion)</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            word_count = len(user_text.split())
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{word_count}</div>
                <div class="metric-lbl">Words Analyzed</div>
            </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=polarity,
            number=dict(suffix="", font=dict(color="#e2e8f0", size=28)),
            gauge=dict(
                axis=dict(range=[-1, 1], tickcolor="#64748b",
                          tickfont=dict(color="#64748b"),
                          tickvals=[-1, -0.5, 0, 0.5, 1],
                          ticktext=["Very\nNegative", "Negative", "Neutral", "Positive", "Very\nPositive"]),
                bar=dict(color=pol_color, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[-1, -0.1], color="rgba(248,113,113,0.15)"),
                    dict(range=[-0.1, 0.1], color="rgba(251,191,36,0.15)"),
                    dict(range=[0.1, 1], color="rgba(52,211,153,0.15)"),
                ],
                threshold=dict(line=dict(color=pol_color, width=3), thickness=0.75, value=polarity),
            ),
            title=dict(text="Sentiment Polarity Gauge", font=dict(color="#94a3b8", size=14)),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit", color="#94a3b8"),
            height=220,
            margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Word highlighting
        st.markdown("#### 🔬 Word Analysis")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:1rem 1.2rem;
                    font-size:1rem; line-height:2; border:1px solid rgba(255,255,255,0.08)">
        """ + highlight_words(user_text.strip()) + """
        </div>
        <div style="margin-top:0.5rem; font-size:0.8rem; color:#64748b">
            <span style="background:rgba(52,211,153,0.2); border-radius:4px; padding:1px 6px; color:#34d399">Green</span> = positive word &nbsp;&nbsp;
            <span style="background:rgba(248,113,113,0.2); border-radius:4px; padding:1px 6px; color:#f87171">Red</span> = negative word
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; border:2px dashed rgba(244,114,182,0.3);
                    border-radius:20px; color:#475569; margin-top:1rem">
            <div style="font-size:3.5rem">✍️</div>
            <div style="font-size:1.1rem; margin-top:0.5rem; color:#64748b">
                Type something and click Analyze!<br>
                <span style="font-size:0.85rem">Try a happy sentence, a sad one, or a fact</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
