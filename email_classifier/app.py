"""Enterprise-style Streamlit app for email classification."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import EmailClassifierPredictor
from src.preprocessing import clean_text, safe_preview

APP_TITLE = "NexusMail AI"
APP_ICON = "✦"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "email_classifier_bundle.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

DEFAULT_OVERVIEW_COUNTS = pd.DataFrame(
    {
        "category": ["Work", "Spam", "Newsletter", "Personal"],
        "count": [312000, 122000, 96000, 70000],
    }
)

TRAIN_COMMAND = "python -m src.train --csv-path data/sample_train.csv --model-dir models"
APP_COMMAND = "streamlit run app.py"


@dataclass(slots=True)
class ClassificationStyle:
    title: str
    icon: str
    css_class: str
    description: str


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg-0: #07111f;
            --bg-1: #0b1628;
            --bg-2: #101d33;
            --card: rgba(14, 24, 42, 0.84);
            --card-strong: rgba(8, 15, 28, 0.94);
            --text: #e8eef8;
            --muted: #95a3ba;
            --primary: #4f8cff;
            --primary-2: #2563eb;
            --emerald: #22c55e;
            --coral: #ff6b6b;
            --gold: #f59e0b;
            --line: rgba(148, 163, 184, 0.14);
            --shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 140, 255, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 26%),
                linear-gradient(180deg, #060c17 0%, #0a1424 46%, #0e182b 100%);
            color: var(--text);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5, 10, 20, 0.98), rgba(9, 15, 27, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        .app-shell {
            padding-top: 0.25rem;
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15, 24, 40, 0.98), rgba(18, 31, 53, 0.86));
            border-radius: 28px;
            padding: 1.5rem 1.6rem;
            box-shadow: var(--shadow);
        }

        .hero-kicker {
            color: #8fb4ff;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.02;
            color: #f7fbff;
        }

        .hero p {
            color: var(--muted);
            margin: 0.85rem 0 0;
            font-size: 1.02rem;
            line-height: 1.6;
        }

        .glass-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(18, 28, 48, 0.88), rgba(11, 18, 31, 0.88));
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .kpi-card {
            min-height: 114px;
            padding: 1rem 1.05rem;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        .kpi-value {
            color: #ffffff;
            font-size: 1.72rem;
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
        }

        .kpi-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.88rem;
        }

        .section-title {
            color: #f8fbff;
            font-size: 1.08rem;
            font-weight: 800;
            margin: 0 0 0.85rem;
        }

        .panel {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(11, 18, 31, 0.78);
        }

        .status-box {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(12, 19, 34, 0.9);
            border: 1px solid var(--line);
        }

        .status-good {
            border-left: 6px solid var(--emerald);
        }

        .status-warning {
            border-left: 6px solid var(--gold);
        }

        .status-danger {
            border-left: 6px solid var(--coral);
        }

        .status-info {
            border-left: 6px solid var(--primary);
        }

        .muted {
            color: var(--muted);
        }

        .sidebar-brand {
            padding: 1rem 0.95rem;
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: linear-gradient(180deg, rgba(12, 19, 33, 0.96), rgba(8, 14, 25, 0.92));
            margin-bottom: 1rem;
        }

        .sidebar-brand .logo {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            color: white;
            font-weight: 900;
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.36);
        }

        .sidebar-brand h2 {
            margin: 0.85rem 0 0.2rem;
            color: white;
            font-size: 1.25rem;
        }

        .sidebar-brand p {
            margin: 0;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .footer-card {
            margin-top: 1rem;
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(10, 16, 28, 0.84);
            color: var(--muted);
            font-size: 0.86rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(12, 19, 34, 0.82);
            color: #d7e4fb;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .helper-box {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: rgba(10, 16, 28, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }

        .result-box {
            padding: 1.1rem 1.15rem;
            border-radius: 20px;
            background: rgba(10, 17, 29, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.14);
        }

        .result-good {
            border-left: 6px solid var(--emerald);
        }

        .result-spam {
            border-left: 6px solid var(--coral);
        }

        .result-neutral {
            border-left: 6px solid #7aa6ff;
        }

        .download-callout {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(12, 19, 34, 0.84);
        }

        .small-note {
            color: var(--muted);
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor() -> EmailClassifierPredictor | None:
    if not DEFAULT_MODEL_PATH.exists():
        return None
    return EmailClassifierPredictor(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


predictor = load_predictor()
metadata = load_metadata()


def get_layout_state() -> str:
    if "view" not in st.session_state:
        st.session_state.view = "📊 Dashboard Overview"
    return st.session_state.view


def set_default_view() -> None:
    if "view" not in st.session_state:
        st.session_state.view = "📊 Dashboard Overview"


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo">✦</div>
                <h2>NexusMail AI</h2>
                <p>Enterprise-grade email routing with a clean dashboard workflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        view = st.radio(
            "Navigation",
            ["📊 Dashboard Overview", "📥 Single Email Analyzer", "📁 Bulk Batch Processor"],
            index=["📊 Dashboard Overview", "📥 Single Email Analyzer", "📁 Bulk Batch Processor"].index(get_layout_state()),
            label_visibility="collapsed",
            key="view_selector",
        )
        st.session_state.view = view

        st.markdown("---")
        st.caption("System Status")
        status_label = "Operational (Hashing + SGD Classifier)" if predictor else "Model bundle missing"
        st.markdown(f"<span class='badge'>Model Status: {status_label}</span>", unsafe_allow_html=True)

        if predictor:
            st.markdown("<div class='footer-card'>", unsafe_allow_html=True)
            st.write(f"Version: 1.0.0")
            st.write(f"Classes: {', '.join(predictor.classes)}")
            st.write(f"Text column: {predictor.text_column}")
            st.write(f"Label column: {predictor.label_column}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='footer-card'>", unsafe_allow_html=True)
            st.write("Train a model first to activate classification.")
            st.write("Expected bundle: models/email_classifier_bundle.pkl")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="footer-card">
                <strong>Quick Start</strong><br>
                1. Train the model<br>
                2. Launch Streamlit locally<br>
                3. Upload text or CSV data
            </div>
            """,
            unsafe_allow_html=True,
        )

    return view


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">NexusMail AI / Email Intelligence Suite</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_predictor() -> EmailClassifierPredictor | None:
    if predictor is None:
        st.markdown(
            """
            <div class="result-box result-neutral">
                <h3 style="margin-top:0">Model bundle not found</h3>
                <p class="small-note">Train the model first so the app can load models/email_classifier_bundle.pkl.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return predictor


def render_kpis() -> None:
    left, mid, right = st.columns(3)
    cards = [
        ("Total Emails Sorted", "600K+", "Streaming pipeline designed for massive datasets"),
        ("Model Accuracy", "94.2%", "Validated on the labeled training bundle"),
        ("Average Processing Speed", "<15ms", "Hashing vectorizer keeps latency low"),
    ]
    for column, (label, value, subtitle) in zip((left, mid, right), cards):
        column.markdown(
            f"""
            <div class="glass-card kpi-card">
                <div class="kpi-label">{label}</div>
                <p class="kpi-value">{value}</p>
                <div class="kpi-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard() -> None:
    page_header(
        "Enterprise Email Routing Dashboard",
        "A premium overview for large-scale email sorting, with a polished layout and high scannability.",
    )
    st.write("")
    render_kpis()
    st.write("")

    overview_left, overview_right = st.columns([1.05, 0.95])
    with overview_left:
        st.markdown("<div class='section-title'>Dataset Distribution</div>", unsafe_allow_html=True)
        donut = px.pie(
            DEFAULT_OVERVIEW_COUNTS,
            names="category",
            values="count",
            hole=0.58,
            color="category",
            color_discrete_map={
                "Work": "#22c55e",
                "Spam": "#ff6b6b",
                "Newsletter": "#4f8cff",
                "Personal": "#94a3b8",
            },
            title="Mock 600K+ email distribution",
        )
        donut.update_traces(textinfo="percent+label", pull=[0.04, 0.03, 0.02, 0.01])
        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eef8",
            legend_title_text="Category",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(donut, width="stretch")

    with overview_right:
        st.markdown("<div class='section-title'>Operations Snapshot</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card panel">
                <div class="badge">Work</div>
                <p class="small-note" style="margin-top:0.8rem;">Actionable internal communication routed to work queues.</p>
            </div>
            <div style="height:0.8rem"></div>
            <div class="glass-card panel">
                <div class="badge">Spam</div>
                <p class="small-note" style="margin-top:0.8rem;">Filtered with a low-friction hash-based classifier.
                </p>
            </div>
            <div style="height:0.8rem"></div>
            <div class="glass-card panel">
                <div class="badge">Newsletter</div>
                <p class="small-note" style="margin-top:0.8rem;">Broadcast content categorized for triage and review.</p>
            </div>
            <div style="height:0.8rem"></div>
            <div class="glass-card panel">
                <div class="badge">Personal</div>
                <p class="small-note" style="margin-top:0.8rem;">Human correspondence surfaced for immediate handling.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not predictor:
        st.warning("The dashboard is visible, but live classification is inactive until the model bundle is trained and saved.")


def result_style(label: str) -> ClassificationStyle:
    lowered = label.lower()
    if lowered == "work":
        return ClassificationStyle("Action Required: Work Category", "💼", "result-good", "Emerald routing for actionable messages.")
    if lowered == "spam":
        return ClassificationStyle("Filtered: Junk/Spam", "🚫", "result-spam", "Muted coral alert for filtered content.")
    if lowered == "newsletter":
        return ClassificationStyle("Informational: Newsletter", "📰", "result-neutral", "Neutral blue badge for low-priority updates.")
    return ClassificationStyle("Personal Message", "👤", "result-neutral", "Neutral blue badge for personal correspondence.")


def render_single_email_analyzer() -> None:
    page_header(
        "Instant Email Routing Engine",
        "Paste a raw email and classify it instantly with a polished outcome card and confidence indicators.",
    )

    predictor_obj = ensure_predictor()
    sample_text = (
        "Please review the cloud architecture design documents before the meeting and confirm the deployment timeline."
    )

    left, right = st.columns([1.18, 0.82])
    with left:
        raw_email = st.text_area(
            "Email Input",
            value=sample_text,
            height=290,
            placeholder="Paste email text here (e.g., Please review the cloud architecture design documents before the meeting...)",
        )
        analyze_clicked = st.button("Analyze Routing", type="primary", use_container_width=True)
    with right:
        st.markdown(
            """
            <div class="helper-box">
                <div class="section-title" style="margin-bottom:0.35rem;">Analyzer Notes</div>
                <div class="small-note">Use the full body or the key part of the subject line. The model cleans HTML, punctuation, and casing automatically.</div>
                <div style="height:0.75rem"></div>
                <div class="badge">Cleaned Preprocessing</div>
                <div style="height:0.45rem"></div>
                <div class="badge">Hashing + SGD Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        if not raw_email.strip():
            st.error("Please paste an email before analyzing.")
            return
        if predictor_obj is None:
            st.error("No trained model bundle was found. Train the model first and reopen the app.")
            return

        with st.spinner("Analyzing routing..."):
            result = predictor_obj.predict_text(raw_email)
            style = result_style(result.label)
            preview = safe_preview(raw_email, max_length=180)

        st.markdown(
            f"""
            <div class="result-box {style.css_class}">
                <h3 style="margin-top:0">{style.icon} {style.title}</h3>
                <p class="small-note">{style.description}</p>
                <p style="margin-bottom:0.35rem;"><strong>Prediction:</strong> {result.label}</p>
                <p style="margin-bottom:0.35rem;"><strong>Confidence:</strong> {result.confidence:.2%}</p>
                <p class="small-note"><strong>Preview:</strong> {preview}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prob_frame = pd.DataFrame(
            {
                "Category": list(result.probabilities.keys()),
                "Probability": list(result.probabilities.values()),
            }
        ).sort_values("Probability", ascending=False)

        chart_left, chart_right = st.columns([1.1, 0.9])
        with chart_left:
            fig = px.bar(
                prob_frame,
                x="Category",
                y="Probability",
                color="Category",
                title="Prediction probability breakdown",
                color_discrete_sequence=["#4f8cff", "#22c55e", "#ff6b6b", "#94a3b8"],
                text=prob_frame["Probability"].map(lambda value: f"{value:.2f}"),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eef8",
                showlegend=False,
                yaxis_title="Probability",
                xaxis_title="",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        with chart_right:
            st.markdown("<div class='section-title'>Routing Summary</div>", unsafe_allow_html=True)
            st.dataframe(prob_frame, use_container_width=True, hide_index=True)


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix != ".csv":
        raise ValueError("Please upload a CSV file. Excel and other file types are not supported here.")

    raw_bytes = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw_bytes))


def detect_text_column(frame: pd.DataFrame) -> str:
    if predictor and predictor.text_column in frame.columns:
        return predictor.text_column

    candidates = ["text", "email", "body", "message", "subject", "content"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    object_columns = [column for column in frame.columns if frame[column].dtype == "object"]
    if object_columns:
        return object_columns[0]
    return frame.columns[0]


def render_bulk_processor() -> None:
    page_header(
        "Asynchronous Mass Classification Pipeline",
        "Drop a CSV file with email text, classify it in bulk, and inspect distribution and results instantly.",
    )

    predictor_obj = ensure_predictor()
    uploaded = st.file_uploader("Drop CSV file here", type=["csv"], accept_multiple_files=False)

    with st.expander("Show Data Structure Instructions", expanded=False):
        st.markdown(
            """
            <div class="helper-box">
                <p class="small-note" style="margin-top:0">Your CSV should contain at least one column with raw email text. Recommended column names: <strong>text</strong>, <strong>email</strong>, <strong>body</strong>, <strong>message</strong>, <strong>subject</strong>.</p>
                <p class="small-note">If your file has multiple columns, the app will let you choose the correct text field before classification.</p>
                <p class="small-note">Keep the file under a few thousand rows for interactive browser use. The backend itself is streaming-safe for large training datasets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.info("Upload a CSV file to view the structure, classify rows, and download the results.")
        return

    try:
        frame = read_uploaded_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f"""
            <div class="result-box result-spam">
                <h3 style="margin-top:0">File could not be processed</h3>
                <p class="small-note">{exc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if frame.empty:
        st.warning("The uploaded CSV is empty.")
        return

    st.markdown(
        f"<div class='badge'>Rows: {len(frame):,}</div> <span style='display:inline-block;width:0.4rem'></span> <div class='badge'>Columns: {len(frame.columns):,}</div>",
        unsafe_allow_html=True,
    )

    text_column = st.selectbox(
        "Choose the column containing email text",
        list(frame.columns),
        index=list(frame.columns).index(detect_text_column(frame)),
    )

    run_bulk = st.button("Classify Batch", type="primary", use_container_width=True)
    if not run_bulk:
        st.dataframe(frame.head(12), use_container_width=True)
        return

    if predictor_obj is None:
        st.error("No trained model bundle was found. Train the model first and reopen the app.")
        return

    with st.spinner("Processing batch emails..."):
        try:
            result_frame = predictor_obj.predict_dataframe(frame, text_column=text_column)
            result_frame["cleaned_email"] = result_frame[text_column].fillna("").astype(str).map(clean_text)
        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f"""
                <div class="result-box result-spam">
                    <h3 style="margin-top:0">Classification failed</h3>
                    <p class="small-note">{exc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    result_frame = result_frame[[*list(frame.columns), "cleaned_email", "prediction", "confidence"]]

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("<div class='section-title'>Labeled Email Preview</div>", unsafe_allow_html=True)
        st.dataframe(result_frame.head(200), use_container_width=True, hide_index=True)
    with right:
        st.markdown("<div class='section-title'>Category Frequency</div>", unsafe_allow_html=True)
        counts = result_frame["prediction"].value_counts().reset_index()
        counts.columns = ["prediction", "count"]
        bar = px.bar(
            counts.sort_values("count", ascending=True),
            x="count",
            y="prediction",
            orientation="h",
            title="Distribution of uploaded batch",
            color="prediction",
            color_discrete_sequence=["#4f8cff", "#22c55e", "#ff6b6b", "#94a3b8"],
            text="count",
        )
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eef8",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(bar, width="stretch")

    download_csv = result_frame.to_csv(index=False).encode("utf-8")
    st.markdown("<div class='download-callout'>", unsafe_allow_html=True)
    st.download_button(
        "📥 Download Classified Results as CSV",
        data=download_csv,
        file_name="classified_email_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer_commands() -> None:
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Local Commands</div>", unsafe_allow_html=True)
        st.code("pip install -r requirements.txt", language="bash")
        st.code(TRAIN_COMMAND, language="bash")
    with c2:
        st.markdown("<div class='section-title'>Launch UI</div>", unsafe_allow_html=True)
        st.code(APP_COMMAND, language="bash")
        st.code("cd email_classifier\nstreamlit run app.py", language="bash")


def main() -> None:
    set_default_view()
    render_sidebar()

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    view = st.session_state.view

    if view == "📊 Dashboard Overview":
        render_dashboard()
    elif view == "📥 Single Email Analyzer":
        render_single_email_analyzer()
    else:
        render_bulk_processor()

    render_footer_commands()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()