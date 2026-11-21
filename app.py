from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wssv_model as wm

st.set_page_config(
    page_title="WSSV Symptom Checker",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# Design tokens & styles
# --------------------------------------------------------------------------------------

NAVY = "#0a2540"
TEAL = "#0e9f9a"
RED = "#e5484d"
GREEN = "#1f9d6b"
AMBER = "#e8a02f"
MUTED = "#5b7083"
GRID = "rgba(10, 37, 64, 0.08)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {
    --navy: #0a2540; --teal: #0e9f9a; --teal-soft: #e3f5f4;
    --red: #e5484d; --red-soft: #fdecec; --green: #1f9d6b; --green-soft: #e6f6ef;
    --amber: #e8a02f; --amber-soft: #fdf3e2; --muted: #5b7083; --line: rgba(10,37,64,0.10);
    --card: #ffffff; --bg: #f4f8fb;
}
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
    font-family: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    border-radius: 24px; padding: 2.2rem 2.4rem 2rem;
    background: radial-gradient(1200px 400px at 85% -10%, rgba(14,159,154,0.55), transparent 60%),
                linear-gradient(135deg, #0a2540 0%, #0d3a5c 55%, #0e6f78 100%);
    color: #fff; box-shadow: 0 18px 40px rgba(10,37,64,0.18);
}
.hero::after {
    content: ""; position: absolute; right: -60px; bottom: -80px; width: 320px; height: 320px;
    border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.10), transparent 70%);
}
.hero-grid { display: flex; gap: 2rem; align-items: center; justify-content: space-between; flex-wrap: wrap; }
.hero-kicker {
    display: inline-flex; gap: .5rem; align-items: center;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
    padding: .3rem .8rem; border-radius: 999px; font-size: .8rem; font-weight: 600; letter-spacing: .02em;
}
.hero h1, .hero h1 * { color: #fff !important; font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin: .8rem 0 .5rem; letter-spacing: -0.02em; }
.hero h1 .accent { color: #7ee3dc !important; }
.hero p { color: rgba(255,255,255,0.82); font-size: 1.04rem; max-width: 640px; margin: 0; }
.hero-badge {
    font-size: 4.6rem; line-height: 1; width: 128px; height: 128px; border-radius: 32px;
    display: grid; place-items: center; background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18); backdrop-filter: blur(6px);
    animation: float 5s ease-in-out infinite;
}
@keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-8px) } }

.stat-row { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: .9rem; margin: 1.1rem 0 1.4rem; }
.stat {
    background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1.1rem;
    display: flex; gap: .8rem; align-items: center; box-shadow: 0 2px 10px rgba(10,37,64,0.04);
}
.stat .ico { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; background: var(--teal-soft); font-size: 1.25rem; }
.stat .num { font-size: 1.35rem; font-weight: 800; color: var(--navy); line-height: 1.1; }
.stat .lbl { font-size: .8rem; color: var(--muted); }
@media (max-width: 900px) { .stat-row { grid-template-columns: repeat(2, minmax(0,1fr)); } .hero h1 { font-size: 1.9rem; } .hero-badge { display: none; } }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: .4rem; background: var(--card); padding: .35rem; border-radius: 14px; border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .55rem 1rem; font-weight: 600; color: var(--muted); height: auto; }
.stTabs [aria-selected="true"] { background: var(--navy) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Cards & headings ---------- */
.card-title { display: flex; align-items: center; gap: .55rem; font-weight: 700; color: var(--navy); font-size: 1.08rem; margin: .1rem 0 .2rem; }
.card-sub { color: var(--muted); font-size: .9rem; margin: 0 0 .8rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .card-title) { background: var(--card); }

/* ---------- Verdict ---------- */
.verdict { border-radius: 18px; padding: 1.2rem 1.3rem; border: 1px solid; margin-bottom: .6rem; }
.verdict .tag { font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .85; }
.verdict .title { font-size: 1.55rem; font-weight: 800; margin: .2rem 0 .3rem; line-height: 1.2; }
.verdict .desc { font-size: .93rem; line-height: 1.5; }
.v-wssv { background: var(--red-soft); border-color: rgba(229,72,77,.35); color: #8f1d22; }
.v-healthy { background: var(--green-soft); border-color: rgba(31,157,107,.35); color: #0f5c3e; }
.v-unsure { background: var(--amber-soft); border-color: rgba(232,160,47,.40); color: #7a4d06; }
.v-empty { background: #fff; border-color: var(--line); color: var(--muted); border-style: dashed; text-align: center; padding: 2.4rem 1.3rem; }

.caution {
    display: flex; gap: .6rem; align-items: flex-start; border-radius: 14px; padding: .75rem .9rem;
    background: var(--amber-soft); border: 1px solid rgba(232,160,47,.4); color: #7a4d06; font-size: .88rem; line-height: 1.45;
}

/* ---------- Highlighted text ---------- */
.hl-box { background: #fbfdfe; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; font-size: 1.05rem; line-height: 2.05; }
.hl { padding: .12rem .3rem; border-radius: 6px; }
.legend { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .8rem; color: var(--muted); margin-top: .5rem; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 4px; margin-right: .35rem; vertical-align: -1px; }

/* ---------- Symptom chips ---------- */
.chips { display: flex; flex-wrap: wrap; gap: .45rem; }
.chip { display: inline-flex; align-items: center; gap: .35rem; padding: .35rem .7rem; border-radius: 999px; font-size: .84rem; font-weight: 600; border: 1px solid; }
.chip-on { background: var(--red-soft); color: #8f1d22; border-color: rgba(229,72,77,.35); }
.chip-neg { background: var(--green-soft); color: #0f5c3e; border-color: rgba(31,157,107,.35); text-decoration: line-through; text-decoration-thickness: 1px; }
.chip-off { background: #f3f6f9; color: #8a9aa9; border-color: var(--line); font-weight: 500; }

/* ---------- Misc ---------- */
.step { display: flex; gap: .8rem; align-items: flex-start; margin-bottom: .8rem; }
.step .n { flex: 0 0 30px; height: 30px; border-radius: 9px; background: var(--navy); color: #fff; display: grid; place-items: center; font-weight: 700; font-size: .9rem; }
.step b { color: var(--navy); }
.footer { margin-top: 2.4rem; text-align: center; color: var(--muted); font-size: .85rem; padding: 1rem; border-top: 1px solid var(--line); }
.stButton > button { border-radius: 12px; font-weight: 600; }
.stTextArea textarea { border-radius: 14px; font-size: 1rem; line-height: 1.55; }
</style>
"""
# Blank lines would end the HTML block in Streamlit's markdown renderer, so drop them.
st.markdown("\n".join(line.strip() for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Model loading (cached for the whole server)
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_system():
    dataset = wm.load_dataset()
    system = wm.train_and_evaluate(dataset)
    return dataset, system


EXAMPLES = {
    "🔴 WSSV case": "White spots on the shell and the shrimp turned reddish. They are weak, stopped feeding and are swimming near the pond edge. Many died within three days.",
    "🟢 Normal": "Shrimps are generally nocturnal and hide in burrows during the day, using their antennae to explore their surroundings.",
    "🟡 Unclear": "Shrimp seem a bit slow today and the water looks cloudy.",
}

SYMPTOM_SENTENCES = {
    "White spots": "White spots or patches on the exoskeleton.",
    "Reddish body": "Pink or reddish discoloration of the body.",
    "Soft shell": "Shells are very soft and loose.",
    "Lethargic": "Shrimp are weak and lethargic.",
    "Not feeding": "Reduced feeding or loss of appetite.",
    "Swimming at edge / surface": "Erratic swimming near the water surface and pond edge.",
    "Sudden deaths": "High mortality within a short period.",
    "Moulting problems": "They cannot moult properly.",
    "Opaque body": "Shrimp's body turns opaque and white.",
}

HIGH, LOW = 0.65, 0.35

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def html_block(markup: str) -> None:
    """Render raw HTML. Strips indentation and blank lines so markdown doesn't treat it as code."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


def card_title(icon: str, title: str, sub: str = "") -> None:
    sub_html = f"<p class='card-sub'>{sub}</p>" if sub else ""
    st.markdown(f"<div class='card-title'>{icon} {title}</div>{sub_html}", unsafe_allow_html=True)


def assess(prob: float, hits: list[wm.SymptomHit]) -> dict[str, str]:
    present = [h for h in hits if not h.negated]
    negated = [h for h in hits if h.negated]
    pct = f"{prob * 100:.0f}%"

    if prob >= HIGH:
        if present:
            return {
                "cls": "v-wssv", "tag": "Screening result", "title": "WSSV likely",
                "desc": f"The model gives a {pct} chance of WSSV, and the text mentions {len(present)} recognised WSSV sign{'s' if len(present) != 1 else ''}. Confirm with a PCR test.",
                "short": "WSSV likely",
            }
        return {
            "cls": "v-unsure", "tag": "Low confidence", "title": "Model says WSSV, but no symptoms found",
            "desc": f"The model gives a {pct} chance of WSSV, but no recognised WSSV sign is described"
            + (" (some are mentioned as absent)" if negated else "")
            + ". The training data makes plain sentences look like WSSV, so treat this with caution.",
            "short": "Uncertain",
        }
    if prob <= LOW:
        if present:
            return {
                "cls": "v-unsure", "tag": "Mixed signals", "title": "Model says not WSSV, but symptoms are described",
                "desc": f"The model gives only a {pct} chance of WSSV, yet the text mentions {len(present)} WSSV sign{'s' if len(present) != 1 else ''}. Monitor the pond and consider testing.",
                "short": "Uncertain",
            }
        return {
            "cls": "v-healthy", "tag": "Screening result", "title": "WSSV unlikely",
            "desc": f"The model gives a {pct} chance of WSSV and no recognised WSSV signs are described.",
            "short": "WSSV unlikely",
        }
    return {
        "cls": "v-unsure", "tag": "Screening result", "title": "Uncertain",
        "desc": f"The model gives a {pct} chance of WSSV, which is too close to call. Add more detail about what you observe.",
        "short": "Uncertain",
    }


def gauge(prob: float) -> go.Figure:
    color = RED if prob >= HIGH else GREEN if prob <= LOW else AMBER
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": NAVY}, "valueformat": ".0f"},
            title={"text": "Chance of WSSV", "font": {"size": 14, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)", "tickfont": {"color": MUTED, "size": 11}},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "#eef3f7",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, LOW * 100], "color": "rgba(31,157,107,0.12)"},
                    {"range": [LOW * 100, HIGH * 100], "color": "rgba(232,160,47,0.14)"},
                    {"range": [HIGH * 100, 100], "color": "rgba(229,72,77,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(t=40, b=10, l=35, r=40), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif"))
    return fig


def highlighted_text(influence: list[tuple[str, float | None]]) -> str:
    values = [abs(v) for _, v in influence if v is not None]
    scale = max(values) if values and max(values) > 0 else 1.0
    parts = []
    for i, (token, value) in enumerate(influence):
        safe = html.escape(token)
        is_punct = not token[0].isalnum()
        sep = "" if (i == 0 or is_punct) else " "
        if value is None or abs(value) < 0.004:
            parts.append(f"{sep}{safe}")
            continue
        alpha = 0.15 + 0.6 * min(abs(value) / scale, 1.0)
        rgb = "229,72,77" if value > 0 else "31,157,107"
        tip = f"{'+' if value > 0 else ''}{value * 100:.1f} points toward {'WSSV' if value > 0 else 'not WSSV'}"
        parts.append(f"{sep}<span class='hl' style='background: rgba({rgb},{alpha:.2f})' title='{tip}'>{safe}</span>")
    return "".join(parts)


def symptom_chips(hits: list[wm.SymptomHit]) -> str:
    by_key = {h.key: h for h in hits}
    chips = []
    ordered = sorted(wm.SYMPTOMS, key=lambda s: (0 if s[0] in by_key and not by_key[s[0]].negated else 1 if s[0] in by_key else 2))
    for key, label, _ in ordered:
        hit = by_key.get(key)
        if hit is None:
            chips.append(f"<span class='chip chip-off'>{html.escape(label)}</span>")
        elif hit.negated:
            chips.append(f"<span class='chip chip-neg' title='Mentioned as absent: “{html.escape(hit.matched_text)}”'>✓ {html.escape(label)}</span>")
        else:
            chips.append(f"<span class='chip chip-on' title='Found: “{html.escape(hit.matched_text)}”'>● {html.escape(label)}</span>")
    return f"<div class='chips'>{''.join(chips)}</div>"


def base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# --------------------------------------------------------------------------------------
# State callbacks
# --------------------------------------------------------------------------------------


def set_text(value: str) -> None:
    """Example buttons: fill the box and check it straight away."""
    st.session_state["symptom_text"] = value
    st.session_state["analysed_text"] = value


def submit_text() -> None:
    st.session_state["analysed_text"] = st.session_state.get("symptom_text", "").strip()


def add_selected_symptoms() -> None:
    chosen = st.session_state.get("symptom_pills") or []
    if not chosen:
        return
    addition = " ".join(SYMPTOM_SENTENCES[c] for c in chosen)
    current = st.session_state.get("symptom_text", "").strip()
    st.session_state["symptom_text"] = f"{current} {addition}".strip()
    st.session_state["symptom_pills"] = []


def clear_text() -> None:
    st.session_state["symptom_text"] = ""
    st.session_state["analysed_text"] = ""


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------


def render_diagnose(dataset: wm.Dataset, system: wm.TrainedSystem) -> None:
    st.session_state.setdefault("symptom_text", "")
    st.session_state.setdefault("analysed_text", "")
    st.session_state.setdefault("history", [])

    model_names = list(system.models)
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(border=True):
            card_title("📝", "Describe what you see in the pond", "Write the shrimp's appearance and behaviour in your own words, then click Check for WSSV.")
            with st.form("symptom_form", border=False, enter_to_submit=True):
                st.text_area(
                    "Symptoms",
                    key="symptom_text",
                    height=150,
                    placeholder="e.g. White spots on the shell, shrimp are lethargic and gathering at the pond edge…",
                    label_visibility="collapsed",
                )
                st.form_submit_button("🔍 Check for WSSV", type="primary", width="stretch", on_click=submit_text)
            st.caption("Tip: press Ctrl + Enter (⌘ + Enter on Mac) in the box to check without clicking. Or try an example:")
            ex_cols = st.columns(len(EXAMPLES) + 1)
            for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
                col.button(label, on_click=set_text, args=(text,), width="stretch", key=f"ex_{label}")
            ex_cols[-1].button("✖ Clear", on_click=clear_text, width="stretch", key="clear_btn")

            st.pills(
                "Or build a description from common signs",
                list(SYMPTOM_SENTENCES),
                selection_mode="multi",
                key="symptom_pills",
            )
            add_col, model_col = st.columns([1, 1.4])
            add_col.button("➕ Add selected signs", on_click=add_selected_symptoms, width="stretch", key="add_signs")
            model_name = model_col.selectbox(
                "Model",
                model_names,
                index=model_names.index(system.best_model),
                format_func=lambda n: f"{n} (best)" if n == system.best_model else n,
                help="The best model is chosen automatically from cross-validation. See the Model performance tab.",
                label_visibility="collapsed",
            )

    text = st.session_state["analysed_text"].strip()
    model = system.models[model_name]

    with right:
        if not text:
            st.markdown(
                "<div class='verdict v-empty'><div style='font-size:2.4rem'>🔬</div>"
                "<div class='title' style='font-size:1.15rem;color:#0a2540'>Waiting for a description</div>"
                "<div class='desc'>Type the symptoms on the left, then click <b>🔍 Check for WSSV</b>. "
                "You can also tap an example or build a description from common signs.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            prob = float(wm.predict_proba(model, [text])[0])
            hits = wm.find_symptoms(text)
            verdict = assess(prob, hits)
            st.markdown(
                f"<div class='verdict {verdict['cls']}'><div class='tag'>{verdict['tag']}</div>"
                f"<div class='title'>{verdict['title']}</div><div class='desc'>{verdict['desc']}</div></div>",
                unsafe_allow_html=True,
            )
            if st.session_state["symptom_text"].strip() != text:
                st.info("The description has changed. Click **🔍 Check for WSSV** to update the result.", icon="✏️")
            with st.container(border=True):
                st.plotly_chart(gauge(prob), width="stretch", config={"displayModeBar": False}, key="gauge")

            history = st.session_state["history"]
            if not history or history[0]["Description"] != text or history[0]["Model"] != model_name:
                history.insert(
                    0,
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Description": text,
                        "Model": model_name,
                        "WSSV probability": round(prob, 3),
                        "Result": verdict["short"],
                    },
                )
                del history[20:]

    if not text:
        return

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        with st.container(border=True):
            card_title("🧠", "Why the model decided this", "Each word is coloured by how much it moved the prediction. Hover a word for the exact effect.")
            influence = wm.word_influence(model, text)
            st.markdown(f"<div class='hl-box'>{highlighted_text(influence)}</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='legend'><span><i style='background:rgba(229,72,77,.6)'></i>Pushes toward WSSV</span>"
                "<span><i style='background:rgba(31,157,107,.6)'></i>Pushes toward not WSSV</span>"
                "<span>Darker = stronger effect</span></div>",
                unsafe_allow_html=True,
            )
    with c2:
        with st.container(border=True):
            card_title("🩺", "WSSV symptom checklist", "Known WSSV signs found in your text. This check uses expert rules, not the model.")
            st.markdown(symptom_chips(hits), unsafe_allow_html=True)
            st.markdown(
                "<div class='legend'><span><i style='background:#fdecec;border:1px solid rgba(229,72,77,.5)'></i>Present</span>"
                "<span><i style='background:#e6f6ef;border:1px solid rgba(31,157,107,.5)'></i>Mentioned as absent</span>"
                "<span><i style='background:#f3f6f9;border:1px solid rgba(10,37,64,.15)'></i>Not mentioned</span></div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        card_title("🤝", "What every model says", "If the models disagree, the description is probably ambiguous.")
        rows = []
        for name, m in system.models.items():
            p = float(wm.predict_proba(m, [text])[0])
            rows.append(
                {
                    "Model": f"{name} ⭐" if name == system.best_model else name,
                    "Chance of WSSV": p,
                    "Verdict": "WSSV" if p >= HIGH else "Not WSSV" if p <= LOW else "Uncertain",
                    "Cross-validated accuracy": system.reports[name].metrics["Accuracy"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Chance of WSSV": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Cross-validated accuracy": st.column_config.NumberColumn(format="percent"),
            },
        )

    if len(st.session_state["history"]) > 1:
        with st.expander(f"🕘 Recent checks this session ({len(st.session_state['history'])})"):
            st.dataframe(
                pd.DataFrame(st.session_state["history"]),
                hide_index=True,
                width="stretch",
                column_config={"WSSV probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
            )


def render_batch(system: wm.TrainedSystem) -> None:
    with st.container(border=True):
        card_title("📋", "Check many descriptions at once", "Paste one description per line, or upload a CSV file with a text column.")
        mode = st.segmented_control("Input", ["Paste text", "Upload CSV"], default="Paste text", label_visibility="collapsed", key="batch_mode")

        texts: list[str] = []
        if mode == "Upload CSV":
            upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
            if upload is not None:
                try:
                    frame = pd.read_csv(upload)
                except UnicodeDecodeError:
                    upload.seek(0)
                    frame = pd.read_csv(upload, encoding="cp1252")
                text_cols = [c for c in frame.columns if frame[c].dtype == object]
                if not text_cols:
                    st.error("No text column was found in this file.")
                    return
                default = next((i for i, c in enumerate(text_cols) if str(c).lower() in {"message", "text", "symptoms", "description"}), 0)
                column = st.selectbox("Which column holds the descriptions?", text_cols, index=default)
                texts = frame[column].dropna().astype(str).tolist()
        else:
            pasted = st.text_area(
                "Descriptions",
                height=170,
                key="batch_text",
                value="White spots on the carapace and sudden mortality\nShrimp are feeding actively and moulting normally\nReddish body, lethargic, gathering at the pond edge\nThey communicate through chemical signals and visual displays",
                label_visibility="collapsed",
            )
            texts = [line.strip() for line in pasted.splitlines() if line.strip()]

        model_names = list(system.models)
        model_name = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(system.best_model),
            format_func=lambda n: f"{n} (best)" if n == system.best_model else n,
            key="batch_model",
        )

    if not texts:
        st.info("Add some descriptions above to see results.")
        return

    texts = texts[:2000]
    probs = wm.predict_proba(system.models[model_name], texts)
    rows = []
    for text, prob in zip(texts, probs):
        hits = wm.find_symptoms(text)
        rows.append(
            {
                "Description": text,
                "Chance of WSSV": float(prob),
                "Result": assess(float(prob), hits)["short"],
                "Signs found": ", ".join(h.label for h in hits if not h.negated) or "None",
            }
        )
    results = pd.DataFrame(rows)

    counts = results["Result"].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Descriptions", len(results))
    m2.metric("🔴 WSSV likely", int(counts.get("WSSV likely", 0)))
    m3.metric("🟡 Uncertain", int(counts.get("Uncertain", 0)))
    m4.metric("🟢 WSSV unlikely", int(counts.get("WSSV unlikely", 0)))

    st.dataframe(
        results,
        hide_index=True,
        width="stretch",
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Chance of WSSV": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )
    st.download_button(
        "⬇️ Download results (CSV)",
        results.assign(**{"Chance of WSSV": results["Chance of WSSV"].round(4)}).to_csv(index=False).encode("utf-8"),
        file_name="wssv_screening_results.csv",
        mime="text/csv",
    )


def render_performance(dataset: wm.Dataset, system: wm.TrainedSystem) -> None:
    reports = system.reports
    best = reports[system.best_model]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", system.best_model)
    k2.metric("Accuracy", f"{best.metrics['Accuracy'] * 100:.1f}%", help=f"Varied by ± {best.metrics_std['Accuracy'] * 100:.1f} percentage points between test splits.")
    k3.metric("F1 score", f"{best.metrics['F1 score']:.3f}")
    k4.metric("ROC AUC", f"{best.metrics['ROC AUC']:.3f}")
    st.caption(f"All scores come from {system.cv_description} on {len(dataset.clean)} cleaned sentences, so every score is measured on sentences the model did not see during training.")

    with st.container(border=True):
        card_title("📊", "Model comparison", "Average score across all test splits. The thin lines show how much scores varied between splits.")
        palette = {"Accuracy": NAVY, "Precision": TEAL, "Recall": "#6c8cd5", "F1 score": RED, "ROC AUC": AMBER}
        fig = go.Figure()
        names = list(reports)
        for metric in wm.METRIC_NAMES:
            fig.add_bar(
                name=metric,
                x=names,
                y=[reports[n].metrics[metric] for n in names],
                error_y=dict(type="data", array=[reports[n].metrics_std[metric] for n in names], thickness=1, width=3, color="rgba(10,37,64,.35)"),
                marker_color=palette[metric],
                hovertemplate="%{x}<br>" + metric + ": %{y:.3f}<extra></extra>",
            )
        fig.update_layout(barmode="group", bargap=0.25, yaxis=dict(range=[0.75, 1.02], tickformat=".0%"))
        st.plotly_chart(base_layout(fig, 380), width="stretch", key="perf_bars")

        table = pd.DataFrame(
            [{"Model": n, **{m: reports[n].metrics[m] for m in wm.METRIC_NAMES}, "About": wm.MODEL_INFO[n]} for n in names]
        ).sort_values("F1 score", ascending=False)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={m: st.column_config.NumberColumn(format="%.3f") for m in wm.METRIC_NAMES},
        )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        with st.container(border=True):
            card_title("🎯", "Confusion matrix", "How the test predictions compare with the true labels.")
            chosen = st.selectbox("Model", names, index=names.index(system.best_model), key="cm_model", label_visibility="collapsed")
            cm = reports[chosen].confusion
            labels = ["Not WSSV", "WSSV"]
            heat = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=[f"Predicted {l}" for l in labels],
                    y=[f"Actual {l}" for l in labels],
                    colorscale=[[0, "#eef6f8"], [1, TEAL]],
                    showscale=False,
                    text=cm,
                    texttemplate="<b>%{text}</b>",
                    textfont=dict(size=22),
                    hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
                )
            )
            heat.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(heat, 320), width="stretch", key="cm")
    with col_b:
        with st.container(border=True):
            card_title("📈", "ROC curves", "Closer to the top-left corner is better. The dashed line is random guessing.")
            roc = go.Figure()
            colors = [RED, NAVY, TEAL, AMBER]
            for color, n in zip(colors, names):
                r = reports[n]
                roc.add_scatter(x=r.roc_fpr, y=r.roc_tpr, mode="lines", name=f"{n} ({r.metrics['ROC AUC']:.3f})", line=dict(color=color, width=2.5, shape="hv"))
            roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(10,37,64,.3)", dash="dash"), showlegend=False, hoverinfo="skip")
            roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
            st.plotly_chart(base_layout(roc, 360), width="stretch", key="roc")

    wrong = system.misclassified.get(system.best_model)
    with st.expander(f"🔍 Sentences the {system.best_model} got wrong in testing ({0 if wrong is None else len(wrong)})"):
        if wrong is None or wrong.empty:
            st.write("None.")
        else:
            st.dataframe(wrong, hide_index=True, width="stretch")


def top_words(clean: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    counts = vec.fit_transform(clean["text"])
    vocab = np.array(vec.get_feature_names_out())
    pos = np.asarray(counts[clean["label"].to_numpy() == 1].sum(axis=0)).ravel() + 0.5
    neg = np.asarray(counts[clean["label"].to_numpy() == 0].sum(axis=0)).ravel() + 0.5
    log_odds = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
    frame = pd.DataFrame({"word": vocab, "log_odds": log_odds, "wssv": pos - 0.5, "not": neg - 0.5})
    return pd.concat([frame.nlargest(n, "log_odds"), frame.nsmallest(n, "log_odds")])


def render_dataset(dataset: wm.Dataset) -> None:
    clean = dataset.clean
    n_pos = int(clean["label"].sum())
    n_neg = len(clean) - n_pos

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows in file", dataset.raw_count)
    k2.metric("Used for training", len(clean))
    k3.metric("🔴 WSSV symptoms", n_pos)
    k4.metric("🟢 Not WSSV", n_neg)

    col_a, col_b = st.columns([1, 1.6], gap="large")
    with col_a:
        with st.container(border=True):
            card_title("⚖️", "Class balance", "The two classes are almost equal, so accuracy is a fair measure.")
            donut = go.Figure(
                go.Pie(
                    labels=["WSSV", "Not WSSV"],
                    values=[n_pos, n_neg],
                    hole=0.62,
                    marker=dict(colors=[RED, GREEN], line=dict(color="#fff", width=3)),
                    textinfo="value+percent",
                    sort=False,
                )
            )
            donut.update_layout(showlegend=True)
            st.plotly_chart(base_layout(donut, 300), width="stretch", key="donut")
    with col_b:
        with st.container(border=True):
            card_title("🔤", "Most distinctive words", "Words used much more in one class than the other (log-odds).")
            words = top_words(clean)
            words = words.sort_values("log_odds")
            bars = go.Figure(
                go.Bar(
                    x=words["log_odds"],
                    y=words["word"],
                    orientation="h",
                    marker_color=[RED if v > 0 else GREEN for v in words["log_odds"]],
                    customdata=np.stack([words["wssv"], words["not"]], axis=1),
                    hovertemplate="<b>%{y}</b><br>WSSV sentences: %{customdata[0]:.0f}<br>Not-WSSV sentences: %{customdata[1]:.0f}<extra></extra>",
                )
            )
            bars.update_layout(xaxis_title="← more in Not WSSV      more in WSSV →", bargap=0.25)
            st.plotly_chart(base_layout(bars, 560), width="stretch", key="words")

    st.markdown(
        "<div class='caution'><span>⚠️</span><div><b>A limitation of this dataset.</b> The WSSV sentences describe symptoms, "
        "but the not-WSSV sentences are mostly general, descriptive facts about shrimp (for example, words like "
        "<i>fascinating</i>, <i>captivating</i> and <i>mesmerizing</i>). They are not observations of healthy ponds or of other diseases. "
        "So the models partly learn writing style: a plain sentence such as <i>“shrimp are eating well”</i> can still look like WSSV to them. "
        "Adding real healthy-pond and other-disease observations is the best way to improve the model.</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.container(border=True):
        card_title("🗂️", "Browse the data")
        f1, f2 = st.columns([1, 2])
        label_filter = f1.segmented_control("Class", ["All", "WSSV", "Not WSSV"], default="All", key="ds_filter")
        search = f2.text_input("Search", placeholder="Search sentences…", label_visibility="collapsed", key="ds_search")
        view = clean.assign(Class=np.where(clean["label"] == 1, "🔴 WSSV", "🟢 Not WSSV"))[["row_id", "text", "Class"]]
        view = view.rename(columns={"row_id": "Row", "text": "Sentence"})
        if label_filter == "WSSV":
            view = view[clean["label"] == 1]
        elif label_filter == "Not WSSV":
            view = view[clean["label"] == 0]
        if search:
            view = view[view["Sentence"].str.contains(search, case=False, regex=False)]
        st.dataframe(view, hide_index=True, width="stretch", height=360, column_config={"Sentence": st.column_config.TextColumn(width="large")})

    if not dataset.removed.empty:
        with st.expander(f"🧹 Rows removed during cleaning ({len(dataset.removed)})"):
            removed = dataset.removed.assign(label=np.where(dataset.removed["label"] == 1, "WSSV", "Not WSSV"))
            st.dataframe(
                removed.rename(columns={"row_id": "Row", "text": "Sentence", "label": "Label", "reason": "Reason"}),
                hide_index=True,
                width="stretch",
            )
            st.caption("Apostrophes saved in a Windows encoding were also fixed. The original file is left unchanged.")


def render_about(system: wm.TrainedSystem) -> None:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        with st.container(border=True):
            card_title("🦠", "What is WSSV?")
            st.markdown(
                "White Spot Syndrome Virus (WSSV) is one of the most damaging viral diseases of farmed shrimp. "
                "It can kill up to 100% of a pond within 3 to 10 days of the first signs. Common signs include white spots on the shell, "
                "a reddish body colour, lethargy, reduced feeding, and shrimp gathering near the surface or pond edge."
            )
        with st.container(border=True):
            card_title("⚠️", "Important")
            st.markdown(
                "This tool is a **screening aid**, not a diagnosis. It reads a written description and estimates how much it sounds like WSSV. "
                "Always confirm suspected cases with a laboratory test such as **PCR**, and follow your local biosecurity advice."
            )
    with col_b:
        with st.container(border=True):
            card_title("⚙️", "How it works")
            steps = [
                ("Clean the data", "Fix text encoding, remove duplicates and one non-symptom row."),
                ("Turn text into numbers", "TF-IDF captures which words, word pairs and letter patterns appear in each sentence."),
                ("Train four models", "Linear SVM, Logistic Regression, Naive Bayes and Random Forest."),
                ("Test fairly", f"{system.cv_description}. The model with the best F1 score becomes the default."),
                ("Explain each result", "Word colouring shows what drove the prediction, and an expert symptom checklist double-checks it."),
            ]
            st.markdown(
                "".join(f"<div class='step'><div class='n'>{i}</div><div><b>{t}</b><br><span style='color:#5b7083'>{d}</span></div></div>" for i, (t, d) in enumerate(steps, 1)),
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            card_title("🧭", "Next steps")
            st.markdown(
                "- Add symptom descriptions for **other shrimp diseases** to turn this into a multi-disease classifier.\n"
                "- Add real **healthy-pond observations** so the model learns symptoms, not writing style."
            )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


def main() -> None:
    html_block(
        """
        <div class="hero"><div class="hero-grid">
            <div>
                <div class="hero-kicker">🧬 AI screening · Shrimp health</div>
                <h1>WSSV <span class="accent">Symptom Checker</span></h1>
                <p>Describe what you see in your pond in plain words. The model estimates whether it points to White Spot Syndrome Virus, and shows you exactly why.</p>
            </div>
            <div class="hero-badge">🦐</div>
        </div></div>
        """
    )

    with st.spinner("Training and testing the models (first load only, about 15 seconds)…"):
        try:
            dataset, system = get_system()
        except FileNotFoundError:
            st.error("Dataset not found. Make sure data/wssv_symptoms.csv sits next to app.py.")
            st.stop()

    best = system.reports[system.best_model]
    html_block(
        f"""
        <div class="stat-row">
            <div class="stat"><div class="ico">📚</div><div><div class="num">{len(dataset.clean)}</div><div class="lbl">Labelled sentences</div></div></div>
            <div class="stat"><div class="ico">🤖</div><div><div class="num">{len(system.models)}</div><div class="lbl">Models compared</div></div></div>
            <div class="stat"><div class="ico">🏆</div><div><div class="num">{system.best_model}</div><div class="lbl">Best model</div></div></div>
            <div class="stat"><div class="ico">🎯</div><div><div class="num">{best.metrics['Accuracy'] * 100:.1f}%</div><div class="lbl">Tested accuracy</div></div></div>
        </div>
        """
    )

    tabs = st.tabs(["🩺 Diagnose", "📋 Batch check", "📊 Model performance", "🗂️ Dataset", "ℹ️ About"])
    with tabs[0]:
        render_diagnose(dataset, system)
    with tabs[1]:
        render_batch(system)
    with tabs[2]:
        render_performance(dataset, system)
    with tabs[3]:
        render_dataset(dataset)
    with tabs[4]:
        render_about(system)

    st.markdown(
        "<div class='footer'>🦐 WSSV Symptom Checker · A screening aid built on labelled symptom descriptions. Confirm suspected cases with a PCR test.</div>",
        unsafe_allow_html=True,
    )


main()
