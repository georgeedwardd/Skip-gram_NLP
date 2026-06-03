import numpy as np
import json
import random
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Word Ascent", page_icon="🔺", layout="centered")

# ── Load artefacts (cached so they're only read once) ──────────────────────────
BASE = Path(__file__).parent.parent

@st.cache_resource
def load_data():
    data = np.load(BASE / "artefacts" / "filtered_embeddings.npz")
    W = data["W"]
    with open(BASE / "artefacts" / "filtered_word_idx.json") as f:
        word_idx = json.load(f)
    with open(BASE / "artefacts" / "filtered_idx_word.json") as f:
        idx_word = {int(k): v for k, v in json.load(f).items()}
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)
    return W_norm, word_idx, idx_word

W_norm, word_idx, idx_word = load_data()

# ── Game logic ─────────────────────────────────────────────────────────────────
MAX_STEPS = 10
FREQ_MIN, FREQ_MAX = 75, 10_000


def get_hint_word(sims, rank_target, exclude=None):
    exclude = set(exclude) if exclude else set()
    sorted_ids = np.argsort(-sims)
    filtered = sorted_ids[np.isin(sorted_ids, np.arange(FREQ_MIN, FREQ_MAX))]
    count = 0
    for idx in filtered:
        word = idx_word[int(idx)]
        if word in exclude:
            continue
        if count == rank_target:
            return word
        count += 1
    return None


def new_game(target=None):
    if target is None:
        target = random.choice(list(word_idx.keys())[FREQ_MIN:FREQ_MAX])
    target_id = word_idx[target]
    target_vec = W_norm[target_id]
    sims = np.dot(W_norm, target_vec)
    opening_hint = get_hint_word(sims, rank_target=50)
    return {
        "target": target,
        "target_id": target_id,
        "target_vec": target_vec,
        "step": 0,
        "guesses": [],        # list of {"word", "rank", "step"}
        "hints": [],          # hint words already revealed (strings)
        "hint_msgs": [],      # full hint display strings
        "opening_hint": opening_hint,
        "status": "playing",  # "playing" | "won" | "lost"
    }


def process_guess(gs, guess):
    """Mutates gs in-place. Returns a result dict."""
    guess = guess.strip().lower()
    if guess not in word_idx:
        return {"error": f"'{guess}' is not in the vocabulary."}

    guessed_words = {g["word"] for g in gs["guesses"]}
    if guess in guessed_words:
        return {"error": f"Already guessed '{guess}'."}
    if guess == gs["opening_hint"] or guess in gs["hints"]:
        return {"error": f"'{guess}' is a hint word and cannot be guessed."}

    gs["step"] += 1
    guess_id = word_idx[guess]

    if guess_id == gs["target_id"]:
        gs["status"] = "won"
        gs["guesses"].append({"word": guess, "rank": 1, "step": gs["step"]})
        return {"rank": 1, "won": True}

    sims = np.dot(W_norm, gs["target_vec"])
    rank = int(np.sum(sims > sims[guess_id]) + 1)
    gs["guesses"].append({"word": guess, "rank": rank, "step": gs["step"]})

    guessed_words.add(guess)
    hint_msg = None

    if gs["step"] == 3:
        hw = get_hint_word(sims, rank_target=10, exclude=guessed_words)
        gs["hints"].append(hw)
        hint_msg = f'Hint (step 3): a related word is <strong>"{hw}"</strong>'
        gs["hint_msgs"].append(hint_msg)

    if gs["step"] == 6:
        hw = get_hint_word(sims, rank_target=5, exclude=guessed_words)
        gs["hints"].append(hw)
        hint_msg = f'Hint (step 6): a closer word is <strong>"{hw}"</strong>'
        gs["hint_msgs"].append(hint_msg)

    if gs["step"] == 9:
        hw = get_hint_word(sims, rank_target=2, exclude=guessed_words)
        gs["hints"].append(hw)
        hint_msg = f'Hint (step 9): a very close word is <strong>"{hw}"</strong>'
        gs["hint_msgs"].append(hint_msg)

    if gs["step"] >= MAX_STEPS:
        gs["status"] = "lost"

    return {"rank": rank, "won": False, "hint_msg": hint_msg,
            "lost": gs["status"] == "lost"}


# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap');

/* Background — cover all Streamlit wrapper layers */
html, body, .stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stMain"],
.main .block-container {
    background-color: #060a0f !important;
    color: #c9d8e4 !important;
}
.block-container { max-width: 560px !important; padding-top: 2rem; }

/* Form background */
[data-testid="stForm"] {
    background-color: #060a0f !important;
    border: none !important;
}

h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: .2em;
    color: #00ffe7 !important;
    text-shadow: 0 0 30px #00ffe7, 0 0 60px rgba(0,255,231,.3);
    font-size: 3rem !important;
    margin-bottom: 0 !important;
}
.subtitle {
    color: #4a6b7c;
    font-family: 'Share Tech Mono', monospace;
    font-size: .75rem;
    letter-spacing: .4em;
    margin-bottom: 1.5rem;
}

/* Progress bar */
.pip-row { display:flex; gap:4px; margin-bottom:1.5rem; }
.pip { flex:1; height:6px; border-radius:2px; background:#1e2d3d; }
.pip.done { background:#00ffe7; }
.pip.danger { background:#ff6b35; }

/* Log entries */
.log-entry {
    display:flex; align-items:center; gap:1rem;
    padding:.5rem .8rem; background:#0d1117;
    border-left:3px solid #1e2d3d;
    margin-bottom:.35rem;
    font-family:'Share Tech Mono', monospace;
}
.log-entry.rank-top  { border-left-color:#39ff14; }
.log-entry.rank-mid  { border-left-color:#00ffe7; }
.log-entry.rank-low  { border-left-color:#4a6b7c; }
.log-entry.rank-bad  { border-left-color:#ff6b35; }
.log-entry.won-entry { border-left-color:#39ff14; background:#0d1f0d; }
.log-word { flex:1; font-size:1rem; color:#fff; }
.log-rank { font-size:.85rem; color:#4a6b7c; }
.log-rank span { color:#00ffe7; }

/* Hint box */
.hint-box {
    padding:.6rem 1rem; background:#0d1117;
    border:1px dashed #ff6b35; color:#ff6b35;
    font-family:'Share Tech Mono', monospace;
    font-size:.8rem; margin-bottom:.4rem;
}

/* Rules box */
.rules-box {
    border:1px solid #1e2d3d; background:#0d1117;
    padding:1.5rem; margin-bottom:1.5rem;
}
.rules-box h2 {
    font-family:'Bebas Neue', sans-serif;
    font-size:1.4rem; letter-spacing:.2em;
    color:#00ffe7; border-bottom:1px solid #1e2d3d;
    padding-bottom:.5rem; margin-bottom:1rem;
}
.rules-box li { font-size:.82rem; line-height:1.7; color:#c9d8e4; margin-bottom:.3rem; }
.rules-note {
    border-left:3px solid #ff6b35; color:#ff6b35;
    font-size:.78rem; line-height:1.6; padding:.6rem .8rem;
    margin-top:1rem;
}

/* Win / lose banner */
.result-banner {
    text-align:center; padding:1.2rem;
    font-family:'Bebas Neue', sans-serif;
    font-size:2.5rem; letter-spacing:.2em;
    margin-bottom:1rem;
}
.result-banner.won  { color:#39ff14; text-shadow:0 0 20px #39ff14; }
.result-banner.lost { color:#ff6b35; text-shadow:0 0 20px #ff6b35; }

/* Override Streamlit input/button chrome */
div[data-testid="stTextInput"] input {
    background:#0d1117 !important; color:#00ffe7 !important;
    border:1px solid #1e2d3d !important;
    font-family:'Share Tech Mono', monospace !important;
    font-size:1rem !important;
    caret-color:#00ffe7;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#00ffe7 !important;
    box-shadow:0 0 10px rgba(0,255,231,.2) !important;
}
.stButton > button {
    background:transparent !important; color:#00ffe7 !important;
    border:1px solid #00ffe7 !important;
    font-family:'Share Tech Mono', monospace !important;
    letter-spacing:.1em;
}
.stButton > button:hover {
    background:#00ffe7 !important; color:#060a0f !important;
}
.stAlert { background:#0d1117 !important; border:1px solid #ff6b35 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ────────────────────────────────────────────────────
if "screen" not in st.session_state:
    st.session_state.screen = "menu"   # "menu" | "game"
if "gs" not in st.session_state:
    st.session_state.gs = None
if "error" not in st.session_state:
    st.session_state.error = ""


# ── Header (shown on all screens) ─────────────────────────────────────────────
st.markdown("<h1>Word Ascent</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">SEMANTIC WORD GUESSER</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MENU SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "menu":
    st.markdown("""
    <div class="rules-box">
      <h2>How to Play</h2>
      <ul>
        <li>A secret target word has been chosen. Your goal is to guess it within <strong>10 attempts</strong>.</li>
        <li>Every word in the vocabulary is ranked by semantic similarity to the target.
            The closer in meaning, the lower the rank — rank <strong>#1</strong> means you found it.</li>
        <li>After each guess you receive a <strong>rank</strong> telling you how close you are.
            Use that signal to zero in on the target.</li>
        <li>At steps 3, 6, and 9 you receive a <strong>hint</strong> —
            a word that is semantically very close to the target.</li>
        <li>At the start of each game you also receive an <strong>opening hint</strong>
            to help you get your bearings.</li>
      </ul>
      <div class="rules-note">
        ⚠ All words are <strong>base forms only</strong> — no plurals, no "-ing", no "-ed".
        If your guess isn't in the vocabulary, you'll be told.
        When in doubt, go with the simplest root form of the word.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  PLAY", use_container_width=True):
        st.session_state.gs = new_game()
        st.session_state.screen = "game"
        st.session_state.error = ""
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# GAME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "game":
    gs = st.session_state.gs

    # ── Progress pips ──────────────────────────────────────────────────────────
    pips_html = '<div class="pip-row">'
    for i in range(MAX_STEPS):
        cls = "pip"
        if i < gs["step"]:
            cls += " danger" if i >= MAX_STEPS - 4 else " done"
        pips_html += f'<div class="{cls}"></div>'
    pips_html += "</div>"
    st.markdown(pips_html, unsafe_allow_html=True)

    # ── Opening hint ───────────────────────────────────────────────────────────
    if gs["opening_hint"]:
        st.markdown(
            f'<div class="hint-box">💡 Starting hint: the word is semantically close to '
            f'<strong>"{gs["opening_hint"]}"</strong></div>',
            unsafe_allow_html=True,
        )

    # ── Result banner (won / lost) ─────────────────────────────────────────────
    if gs["status"] == "won":
        steps = gs["step"]
        st.markdown(
            f'<div class="result-banner won">YOU GOT IT — {steps} step{"s" if steps != 1 else ""}!</div>',
            unsafe_allow_html=True,
        )
    elif gs["status"] == "lost":
        st.markdown(
            f'<div class="result-banner lost">GAME OVER — the word was <strong>{gs["target"]}</strong></div>',
            unsafe_allow_html=True,
        )

    # ── Input row (only while playing) ────────────────────────────────────────
    if gs["status"] == "playing":
        with st.form("guess_form", clear_on_submit=True):
            col_input, col_btn = st.columns([4, 1])
            with col_input:
                guess_input = st.text_input(
                    "guess", label_visibility="collapsed",
                    placeholder="enter a word…",
                    key="guess_field",
                )
            with col_btn:
                submitted = st.form_submit_button("GUESS")

        if submitted and guess_input.strip():
            result = process_guess(gs, guess_input)
            if "error" in result:
                st.session_state.error = result["error"]
            else:
                st.session_state.error = ""
            st.rerun()

        if st.session_state.error:
            st.markdown(
                f'<div style="color:#ff6b35;font-size:.85rem;margin-bottom:.5rem;">'
                f'⚠ {st.session_state.error}</div>',
                unsafe_allow_html=True,
            )

    # ── Play again button ──────────────────────────────────────────────────────
    if gs["status"] != "playing":
        if st.button("↺  PLAY AGAIN", use_container_width=True):
            st.session_state.gs = new_game()
            st.session_state.error = ""
            st.rerun()

    # ── Guess log (newest first) ───────────────────────────────────────────────
    def rank_class(r):
        if r <= 50:    return "rank-top"
        if r <= 500:   return "rank-mid"
        if r <= 2000:  return "rank-low"
        return "rank-bad"

    log_html = ""

    # Interleave hint messages at the step they appeared
    hint_step_map = {}
    raw_steps = [3, 6, 9]
    for i, msg in enumerate(gs["hint_msgs"]):
        hint_step_map[raw_steps[i]] = msg

    for entry in reversed(gs["guesses"]):
        # Show hint that fired on this step (appears above the guess it follows)
        if entry["step"] in hint_step_map:
            log_html += (
                f'<div class="hint-box">💡 {hint_step_map[entry["step"]]}</div>'
            )
        cls = "won-entry" if entry["rank"] == 1 else rank_class(entry["rank"])
        log_html += (
            f'<div class="log-entry {cls}">'
            f'<span class="log-word">{entry["word"]}</span>'
            f'<span class="log-rank">rank <span>#{entry["rank"]}</span></span>'
            f'</div>'
        )

    if log_html:
        st.markdown(log_html, unsafe_allow_html=True)