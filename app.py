import re
import textwrap
from io import BytesIO
from typing import Literal, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import arxiv
import streamlit as st
from pypdf import PdfReader


try:
    import openai
except ImportError:  # pragma: no cover - handled at runtime
    openai = None  # type: ignore

try:
    import anthropic
except ImportError:  # pragma: no cover - handled at runtime
    anthropic = None  # type: ignore


Provider = Literal["OpenAI", "Anthropic"]


def set_page_config() -> None:
    st.set_page_config(
        page_title="Research Paper Translator for Public Media",
        page_icon="🧠",
        layout="wide",
    )

    dark_css = """
    <style>
    html, body {
        margin: 0;
        padding: 0;
        background-color: #050510;
    }
    :root, .stApp {
        background-color: #050510;
        color: #f5f5f5;
    }
    .stApp {
        background: radial-gradient(circle at top, #1e293b 0, #020617 40%, #000000 100%);
    }
    /* Remove default white header / toolbar strip */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    header[data-testid="stHeader"] > div {
        display: none;
    }
    .main > div {
        padding-top: 1rem;
    }
    .stSidebar {
        background: linear-gradient(180deg, #020617 0%, #020617 40%, #000000 100%);
        color: #e5e7eb;
    }
    .stSidebar, .stSidebar * {
        color: #e5e7eb !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .neuro-title {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #e5e7eb;
    }
    .neuro-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    .neuro-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.8);
        color: #e5e7eb;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .neuro-tagline {
        font-size: 0.8rem;
        color: #d1d5db;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .neuro-card {
        border-radius: 0.8rem;
        padding: 1.1rem 1.2rem;
        background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.08), rgba(15, 23, 42, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .neuro-meta {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .neuro-meta strong {
        color: #e5e7eb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.15rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 0.9rem;
        background-color: rgba(15, 23, 42, 0.8);
        color: #9ca3af;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e5e7eb !important;
        border-color: rgba(59, 130, 246, 0.6) !important;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.3);
    }
    .neuro-download-col {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-top: 0.4rem;
    }
    .neuro-footer {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(31, 41, 55, 0.9);
        padding-top: 0.7rem;
        text-align: right;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)


def extract_arxiv_id(url: str) -> Optional[str]:
    """
    Extract the arXiv ID from a typical arxiv.org URL.
    Supports /abs/<id>, /pdf/<id>.pdf and variants.
    """
    try:
        parsed = urlparse(url.strip())
        if "arxiv.org" not in parsed.netloc:
            return None
        path = parsed.path
        # Examples:
        # /abs/2401.12345, /pdf/2401.12345.pdf, /abs/2401.12345v2
        m = re.search(r"/(abs|pdf)/([0-9]+\.[0-9]+[v0-9]*)", path)
        if m:
            return m.group(2)
        # Fallback: last path part
        last = path.rstrip("/").split("/")[-1]
        last = last.replace(".pdf", "")
        if re.match(r"^[0-9]+\.[0-9]+(v[0-9]+)?$", last):
            return last
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def fetch_paper_metadata(arxiv_url: str) -> Tuple[str, str]:
    paper_id = extract_arxiv_id(arxiv_url)
    if not paper_id:
        raise ValueError("Could not parse a valid arXiv ID from the provided URL.")

    search = arxiv.Search(id_list=[paper_id])
    try:
        result = next(search.results())
    except StopIteration:
        raise ValueError("No paper found for that arXiv ID.")

    title = result.title.strip()
    pdf_url = result.pdf_url
    return title, pdf_url


@st.cache_data(show_spinner=False)
def download_and_extract_pdf_text(pdf_url: str) -> str:
    with urlopen(pdf_url) as resp:
        pdf_bytes = resp.read()

    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue
    raw_text = "\n".join(pages_text)
    # Light normalisation to help stay within context windows
    return textwrap.dedent(raw_text).strip()


def build_system_prompt() -> str:
    return (
        "You are a senior neuro-AI researcher and science communicator. "
        "Your job is to read an AI / ML paper and surface the NEUROBIOLOGICAL "
        "GAPS between contemporary ML techniques and what is known from systems "
        "neuroscience, cognitive science, and biological learning.\n\n"
        "Definition: 'Neurobiological AI Gaps' are the concrete mismatches between "
        "how a model learns/represents/optimises and how biological brains do so "
        "along dimensions such as: data regime, objective functions, architectures, "
        "inductive biases, learning rules, embodiment, energy constraints, temporal "
        "dynamics, wiring constraints, and adaptation.\n\n"
        "Given the paper text, you must produce THREE artefacts:\n"
        "1) LINKEDIN_CAROUSEL: 5 slides: Hook, The Gap, The Tech, The Neuro-Insight, CTA.\n"
        "2) SUBSTACK_DEEP_DIVE: long-form, technically literate essay.\n"
        "3) RESEARCHER_SIGNAL: a single high-signal X (Twitter) post aimed at researchers.\n\n"
        "Respond EXACTLY in the following tagged format:\n"
        "=== LINKEDIN_CAROUSEL ===\n"
        "Slide 1 - Hook: ...\n"
        "Slide 2 - The Gap: ...\n"
        "Slide 3 - The Tech: ...\n"
        "Slide 4 - The Neuro-Insight: ...\n"
        "Slide 5 - CTA: ...\n"
        "=== SUBSTACK_DEEP_DIVE ===\n"
        "<multi-paragraph deep dive>\n"
        "=== RESEARCHER_SIGNAL ===\n"
        "<single X post, 240 characters or less, no hashtags at the end>\n"
    )


def build_user_prompt(paper_title: str, paper_text: str) -> str:
    head = textwrap.dedent(
        f"""
        You are analysing the following paper:

        TITLE: {paper_title}

        Below is a noisy text extraction of the paper's PDF. It may contain broken formatting.
        Work around this as best you can and focus on the core ideas and methods.

        PAPER TEXT:
        """
    )
    return head + "\n" + paper_text


def call_openai(api_key: str, system_prompt: str, user_prompt: str) -> str:
    if openai is None:
        raise RuntimeError("The 'openai' package is not installed.")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content or ""


def call_anthropic(api_key: str, system_prompt: str, user_prompt: str) -> str:
    if anthropic is None:
        raise RuntimeError("The 'anthropic' package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=0.6,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Anthropics' content is a list of blocks; assume single text block
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


def parse_sections(raw: str) -> Tuple[str, str, str]:
    linked_in = ""
    substack = ""
    signal = ""

    current = None
    lines = raw.splitlines()
    for line in lines:
        tag = line.strip()
        if tag == "=== LINKEDIN_CAROUSEL ===":
            current = "li"
            continue
        if tag == "=== SUBSTACK_DEEP_DIVE ===":
            current = "sub"
            continue
        if tag == "=== RESEARCHER_SIGNAL ===":
            current = "sig"
            continue
        if current == "li":
            linked_in += line + "\n"
        elif current == "sub":
            substack += line + "\n"
        elif current == "sig":
            signal += line + "\n"

    return linked_in.strip(), substack.strip(), signal.strip()


def sidebar_controls() -> Tuple[Optional[Provider], Optional[str]]:
    with st.sidebar:
        st.markdown(
            "<div class='neuro-badge'>Research Paper Translator</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='neuro-tagline'>Translate ArXiv into public narratives</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        provider: Provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"])
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            help=f"Your {provider} key is used only in this session.",
        )

        st.caption(
            "Keys are kept in memory for this session only and are never logged "
            "to disk by this app."
        )

    if not api_key:
        return provider, None
    return provider, api_key.strip()


def main() -> None:
    set_page_config()

    provider, api_key = sidebar_controls()

    # Push main hero content toward the vertical center
    st.markdown("<div style='height:18vh'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='neuro-title'>Research Paper Translator for Public Media</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='neuro-subtitle'>Paste an arXiv link → get public-ready narratives.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Left-align the search bar under the header
    with st.container():
        input_col, button_col, _ = st.columns([4, 1, 1])
        with input_col:
            arxiv_url = st.text_input(
                "ArXiv URL",
                placeholder="https://arxiv.org/abs/2401.12345",
            )
        with button_col:
            run_btn = st.button(
                "Analyse Paper",
                type="primary",
                use_container_width=True,
            )

    if run_btn:
        if not arxiv_url.strip():
            st.error("Please paste a valid arXiv URL.")
            return
        if not api_key:
            st.error("Please provide an API key in the sidebar.")
            return

        with st.spinner("Fetching paper metadata from arXiv..."):
            try:
                paper_title, pdf_url = fetch_paper_metadata(arxiv_url)
            except Exception as e:
                st.error(f"Error fetching paper: {e}")
                return

        st.markdown(
            "<div class='neuro-card'>"
            f"<div class='neuro-meta'><strong>Title:</strong> {paper_title}</div>"
            f"<div class='neuro-meta'><strong>PDF:</strong> <a href='{pdf_url}' target='_blank'>{pdf_url}</a></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Downloading PDF and extracting text..."):
            try:
                paper_text = download_and_extract_pdf_text(pdf_url)
            except Exception as e:
                st.error(f"Error handling PDF: {e}")
                return

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(paper_title, paper_text)

        with st.spinner(f"Calling {provider} and synthesising narratives..."):
            try:
                if provider == "OpenAI":
                    raw = call_openai(api_key, system_prompt, user_prompt)
                else:
                    raw = call_anthropic(api_key, system_prompt, user_prompt)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                return

        li_text, substack_text, signal_text = parse_sections(raw)

        tab1, tab2, tab3 = st.tabs(
            ["LinkedIn Carousel", "Substack Deep-Dive", "Researcher Signal"]
        )

        with tab1:
            st.markdown("#### LinkedIn Carousel: Neurobiological AI Gaps")
            st.markdown(li_text or raw)
            st.markdown("")
            st.download_button(
                "Download LinkedIn Carousel",
                data=(li_text or raw),
                file_name="linkedin_carousel.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with tab2:
            st.markdown("#### Substack Deep-Dive")
            st.markdown(substack_text or raw)
            st.markdown("")
            st.download_button(
                "Download Substack Draft",
                data=(substack_text or raw),
                file_name="substack_deep_dive.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with tab3:
            st.markdown("#### Researcher Signal (X Post)")
            st.markdown(signal_text or raw)
            st.markdown("")
            st.download_button(
                "Download Researcher Signal",
                data=(signal_text or raw),
                file_name="researcher_signal.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown(
            "<div class='neuro-footer'>Built for rapid neuro-informed comms. "
            "Validate any claims against the original paper.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

