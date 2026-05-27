"""Streamlit UI for the bulk email verification tool."""
from __future__ import annotations

import io
import uuid
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.validator import ProcessingProgress, VerificationSummary, process_csv

APP_TITLE = "Bulk Email Verifier (600k+ Capacity)"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WORKING_OUTPUT = DATA_DIR / "working_emails.csv"
INVALID_OUTPUT = DATA_DIR / "invalid_emails.csv"
DEFAULT_COLUMN_NAME = "email"

st.set_page_config(page_title=APP_TITLE, page_icon="@", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        :root {
            --bg0: #08111f;
            --bg1: #0d1728;
            --card: rgba(14, 23, 40, 0.86);
            --card-strong: rgba(8, 14, 25, 0.96);
            --text: #e8eef8;
            --muted: #96a4ba;
            --indigo: #4f8cff;
            --emerald: #22c55e;
            --coral: #ff6b6b;
            --line: rgba(148, 163, 184, 0.14);
            --shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 140, 255, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 24%),
                linear-gradient(180deg, #06101d 0%, #0b1628 44%, #101d33 100%);
            color: var(--text);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5, 10, 20, 0.98), rgba(9, 15, 27, 0.98));
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15, 24, 40, 0.98), rgba(18, 31, 53, 0.86));
            border-radius: 28px;
            padding: 1.5rem 1.6rem;
            box-shadow: var(--shadow);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            line-height: 1.02;
            color: #f7fbff;
        }

        .hero p {
            color: var(--muted);
            margin: 0.85rem 0 0;
            font-size: 1.01rem;
            line-height: 1.6;
        }

        .glass-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(18, 28, 48, 0.88), rgba(11, 18, 31, 0.88));
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .kpi-card {
            min-height: 110px;
            padding: 1rem 1.05rem;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        .kpi-value {
            color: #ffffff;
            font-size: 1.55rem;
            line-height: 1.1;
            margin: 0;
            font-weight: 800;
        }

        .kpi-subtitle {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.88rem;
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

        .panel {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(11, 18, 31, 0.78);
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
            background: linear-gradient(135deg, var(--indigo), #7c3aed);
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

        .result-box {
            padding: 1.05rem 1.1rem;
            border-radius: 20px;
            background: rgba(10, 17, 29, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.14);
        }

        .result-good {
            border-left: 6px solid var(--emerald);
        }

        .result-bad {
            border-left: 6px solid var(--coral);
        }

        .result-neutral {
            border-left: 6px solid var(--indigo);
        }

        .instructions {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">@</div>
            <h2>Bulk Email Verifier</h2>
            <p>Regex + MX validation for large CSV and Excel batches.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<span class='badge'>Model Status: Operational</span>", unsafe_allow_html=True)
    st.sidebar.caption("Validation mode: Syntax check + MX lookup only. No SMTP pinging.")
    st.sidebar.markdown(
        """
        <div class="footer-card">
            Version: 1.0.0<br>
            Stack: pandas + aiodns/dnspython + Streamlit<br>
            Capacity: 600K+ rows via chunked processing
        </div>
        """,
        unsafe_allow_html=True,
    )


def _save_uploaded_csv(uploaded_file: object) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"upload_{uuid.uuid4().hex}.csv"
    input_path = UPLOAD_DIR / unique_name
    input_path.write_bytes(uploaded_file.getvalue())
    return input_path


def _prepare_uploaded_input(uploaded_file: Any) -> Path:
    """Store the upload as a CSV file so the validator can stream it chunk by chunk."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        frame = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
        temp_file = NamedTemporaryFile(delete=False, suffix=".csv", dir=UPLOAD_DIR)
        temp_path = Path(temp_file.name)
        temp_file.close()
        frame.to_csv(temp_path, index=False)
        return temp_path
    return _save_uploaded_csv(uploaded_file)


def _frame_to_excel_bytes(frame: pd.DataFrame, sheet_name: str) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()


def _load_csv_preview(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=25)


def _format_metric_card(label: str, value: str, subtitle: str) -> str:
    return f"""
    <div class="glass-card kpi-card">
        <div class="kpi-label">{label}</div>
        <p class="kpi-value">{value}</p>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def _summary_cards(summary: VerificationSummary) -> None:
    left, right = st.columns(2)
    left.markdown(
        _format_metric_card(
            "Total Working Emails",
            f"{summary.working_count:,}",
            "Syntax valid and MX verified",
        ),
        unsafe_allow_html=True,
    )
    right.markdown(
        _format_metric_card(
            "Total Invalid Emails",
            f"{summary.invalid_count:,}",
            "Malformed or missing MX records",
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    _render_sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>Bulk Email Verifier (600k+ Capacity)</h1>
            <p>Upload a CSV or Excel file, choose the email column, and split the dataset into working and invalid outputs using fast chunked processing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    top_left, top_mid, top_right = st.columns(3)
    top_left.markdown(
        _format_metric_card("Processing Mode", "Chunked", "Built for large CSV files"),
        unsafe_allow_html=True,
    )
    top_mid.markdown(
        _format_metric_card("Validation", "Regex + MX", "No SMTP pinging"),
        unsafe_allow_html=True,
    )
    top_right.markdown(
        _format_metric_card("Output Files", "2 CSVs", "Working and invalid"),
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)
    email_column = st.text_input("Email column name", value=DEFAULT_COLUMN_NAME)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Show Data Structure Instructions", expanded=False):
        st.markdown(
            """
            <div class="panel">
                <p class="instructions">
                    Your CSV or Excel file should contain one column with email addresses. If your file uses a different name,
                    type it in the email column field above. Example accepted names: <strong>email</strong>,
                    <strong>email_address</strong>, <strong>contact_email</strong>, <strong>mail</strong>.
                </p>
                <p class="instructions">
                    The tool validates format first, then checks whether the domain has MX records. That makes it fast,
                    free, and safe for large datasets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    start_clicked = st.button("Start Verification", type="primary", width="stretch")

    result_summary: VerificationSummary | None = st.session_state.get("verification_summary")
    working_preview_path: Path | None = st.session_state.get("working_output_path")
    invalid_preview_path: Path | None = st.session_state.get("invalid_output_path")

    if start_clicked:
        if uploaded is None:
            st.markdown(
                """
                <div class="result-box result-bad">
                    <h3 style="margin-top:0">Missing file</h3>
                    <p class="small-note">Please upload a CSV file before starting verification.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        try:
            input_path = _prepare_uploaded_input(uploaded)
            st.session_state["uploaded_input_path"] = input_path

            progress_bar = st.progress(0)
            status_area = st.empty()

            def _update_progress(progress: ProcessingProgress) -> None:
                progress_bar.progress(progress.progress_ratio)
                status_area.info(
                    f"Chunk {progress.chunk_index}/{progress.chunks_total} | "
                    f"Processed {progress.rows_processed:,}/{progress.total_rows:,} | "
                    f"Working {progress.working_count:,} | Invalid {progress.invalid_count:,}"
                )

            summary = process_csv(
                input_path=input_path,
                output_working=WORKING_OUTPUT,
                output_invalid=INVALID_OUTPUT,
                email_column=email_column.strip() or DEFAULT_COLUMN_NAME,
                progress_callback=_update_progress,
            )

            st.session_state["verification_summary"] = summary
            st.session_state["working_output_path"] = Path(summary.output_working)
            st.session_state["invalid_output_path"] = Path(summary.output_invalid)
            result_summary = summary
            working_preview_path = Path(summary.output_working)
            invalid_preview_path = Path(summary.output_invalid)

            progress_bar.progress(1.0)
            status_area.success(
                f"Completed {summary.processed_chunks} chunks and processed {summary.total_rows:,} rows."
            )

        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f"""
                <div class="result-box result-bad">
                    <h3 style="margin-top:0">Verification failed</h3>
                    <p class="small-note">{exc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    if result_summary is not None:
        st.write("")
        _summary_cards(result_summary)

        st.write("")
        if working_preview_path and working_preview_path.exists():
            working_frame = pd.read_csv(working_preview_path)
            csv_left, xlsx_left = st.columns(2)
            csv_left.download_button(
                "Download working_emails.csv",
                data=working_preview_path.read_bytes(),
                file_name="working_emails.csv",
                mime="text/csv",
                width="stretch",
            )
            xlsx_left.download_button(
                "Download working_emails.xlsx",
                data=_frame_to_excel_bytes(working_frame, "working_emails"),
                file_name="working_emails.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        if invalid_preview_path and invalid_preview_path.exists():
            invalid_frame = pd.read_csv(invalid_preview_path)
            csv_right, xlsx_right = st.columns(2)
            csv_right.download_button(
                "Download invalid_emails.csv",
                data=invalid_preview_path.read_bytes(),
                file_name="invalid_emails.csv",
                mime="text/csv",
                width="stretch",
            )
            xlsx_right.download_button(
                "Download invalid_emails.xlsx",
                data=_frame_to_excel_bytes(invalid_frame, "invalid_emails"),
                file_name="invalid_emails.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

        st.write("")
        preview_left, preview_right = st.columns(2)
        with preview_left:
            st.markdown("<div class='panel'><strong>Working preview</strong></div>", unsafe_allow_html=True)
            if working_preview_path and working_preview_path.exists():
                st.dataframe(_load_csv_preview(working_preview_path), width="stretch")
        with preview_right:
            st.markdown("<div class='panel'><strong>Invalid preview</strong></div>", unsafe_allow_html=True)
            if invalid_preview_path and invalid_preview_path.exists():
                st.dataframe(_load_csv_preview(invalid_preview_path), width="stretch")

    elif uploaded is not None:
        st.info("Upload a CSV and click Start Verification to generate the working and invalid files.")


if __name__ == "__main__":
    main()
