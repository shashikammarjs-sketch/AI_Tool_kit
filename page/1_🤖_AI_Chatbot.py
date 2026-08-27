import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="AI Chatbot | AI Lab", page_icon="🤖", layout="wide")

# Auto-load API key from secrets if available
if "api_key" not in st.session_state:
    try:
        st.session_state.api_key = st.secrets["gemini"]["api_key"]
    except Exception:
        pass

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

.page-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }

.concept-box {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    color: #c4b5fd; font-size: 0.9rem; line-height: 1.6;
}
.concept-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; color: #a78bfa; }

.user-bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1.1rem; margin: 0.4rem 0 0.4rem auto;
    max-width: 75%; color: #fff; font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3);
}
.bot-bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px 18px 18px 4px;
    padding: 0.7rem 1.1rem; margin: 0.4rem auto 0.4rem 0;
    max-width: 78%; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.chat-label { font-size: 0.72rem; color: #64748b; margin: 2px 4px; }

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    color: #f1f5f9 !important; border-radius: 12px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🤖 AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Chat with an AI assistant and learn how Large Language Models work!</div>', unsafe_allow_html=True)

main_col, sidebar_col = st.columns([2, 1], gap="large")

# ── Sidebar: concepts ──────────────────────────────────────────────────────────
with sidebar_col:
    st.markdown("""
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
    """, unsafe_allow_html=True)

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

# ── Main: API Key + Chat ───────────────────────────────────────────────────────
with main_col:
    # API Key input — collapsed if already loaded from secrets
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
            if st.button("✅ Connect", use_container_width=True) and api_key_input:
                st.session_state.api_key = api_key_input
                st.session_state.messages = []
                st.success("Connected! Start chatting below 🎉")
                st.rerun()
        with col_b:
            if st.button("🎭 Demo Mode (No Key)", use_container_width=True):
                st.session_state.api_key = "DEMO"
                st.session_state.messages = []
                st.info("Demo mode: canned responses only.")
                st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # System persona
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
        try:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=SYSTEM,
            )
            history = []
            for msg in st.session_state.messages[:-1]:  # exclude last (current) user msg
                history.append({"role": msg["role"], "parts": [msg["content"]]})
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error: {e}\n\nPlease check your API key and try again."

    # Handle starter button injection
    if "starter_q" in st.session_state:
        starter = st.session_state.pop("starter_q")
        st.session_state.messages.append({"role": "user", "content": starter})
        if "api_key" in st.session_state:
            reply = get_ai_response(starter)
            st.session_state.messages.append({"role": "model", "content": reply})

    # Render chat history
    if "api_key" in st.session_state:
        chat_container = st.container(height=480)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center; padding:3rem; color:#475569;">
                    <div style="font-size:3rem">👋</div>
                    <div style="font-size:1.1rem; margin-top:0.5rem">Hi! I'm <strong style="color:#a78bfa">Aria</strong>,
                    your AI tutor.<br>Ask me anything about science, math, or any school subject!</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-label" style="text-align:right">You</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-label">🤖 Aria</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

        # Input row
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
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Aria is thinking…"):
                reply = get_ai_response(user_input.strip())
            st.session_state.messages.append({"role": "model", "content": reply})
            st.rerun()

        # Clear button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("👆 Please enter your Gemini API key or use Demo Mode to start chatting!")
