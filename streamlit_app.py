import base64
import os
import re
import shutil
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


load_dotenv()

APP_TITLE = "SOLVAI"
APP_TAGLINE = "Vos problemes repetitifs. Notre IA. Resolus."
MAX_FILE_SIZE_MB = 10
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180


@dataclass
class Chunk:
    document_name: str
    source_label: str
    text: str


def ensure_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "workspace_dir" not in st.session_state:
        workspace_dir = Path(tempfile.gettempdir()) / f"solvai_demo_{st.session_state.session_id}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.workspace_dir = str(workspace_dir)


def reset_session() -> None:
    workspace = Path(st.session_state.get("workspace_dir", ""))
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    for key in ["chunks", "documents", "workspace_dir", "session_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def get_openai_client() -> Optional["OpenAI"]:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAI:
        return OpenAI(api_key=api_key)
    return None


def get_anthropic_client() -> Optional["Anthropic"]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and Anthropic:
        return Anthropic(api_key=api_key)
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + CHUNK_SIZE, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - CHUNK_OVERLAP, 0)
    return chunks


def save_uploaded_file(uploaded_file) -> Path:
    workspace = Path(st.session_state.workspace_dir)
    destination = workspace / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def extract_pdf_chunks(file_path: Path) -> List[Chunk]:
    chunks = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            for piece in chunk_text(page_text):
                chunks.append(
                    Chunk(
                        document_name=file_path.name,
                        source_label=f"{file_path.name} · page {page_index}",
                        text=piece,
                    )
                )
    return chunks


def image_to_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def extract_image_text(file_path: Path) -> str:
    client = get_openai_client()
    if not client:
        raise RuntimeError("OCR indisponible sans OPENAI_API_KEY.")
    mime_type = Image.open(file_path).get_format_mimetype()
    image_b64 = image_to_base64(file_path)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract all readable business text from this image. "
                            "Return plain text only, preserving tables and labels when possible."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_b64}",
                    },
                ],
            }
        ],
    )
    return normalize_text(getattr(response, "output_text", ""))


def extract_audio_text(file_path: Path) -> str:
    client = get_openai_client()
    if not client:
        raise RuntimeError("Transcription indisponible sans OPENAI_API_KEY.")
    with file_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return normalize_text(getattr(transcript, "text", ""))


def score_chunk(question: str, chunk: Chunk) -> float:
    question_terms = set(re.findall(r"\w+", question.lower()))
    chunk_terms = set(re.findall(r"\w+", chunk.text.lower()))
    if not question_terms or not chunk_terms:
        return 0.0
    overlap = len(question_terms & chunk_terms)
    coverage = overlap / max(len(question_terms), 1)
    density = overlap / max(len(chunk_terms), 1)
    return (coverage * 0.8) + (density * 0.2)


def retrieve_top_chunks(question: str, limit: int = 4) -> List[Chunk]:
    ranked = sorted(
        st.session_state.chunks,
        key=lambda chunk: score_chunk(question, chunk),
        reverse=True,
    )
    return [chunk for chunk in ranked[:limit] if score_chunk(question, chunk) > 0]


def build_context(chunks: List[Chunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[Source {idx}] {chunk.source_label}\n{chunk.text}")
    return "\n\n".join(lines)


def answer_with_anthropic(question: str, chunks: List[Chunk]) -> str:
    client = get_anthropic_client()
    if not client:
        raise RuntimeError("ANTHROPIC_API_KEY absent.")
    prompt = textwrap.dedent(
        f"""
        Tu reponds uniquement a partir des extraits fournis.
        Si l'information n'est pas clairement presente, dis-le.
        Reponds en francais, de maniere concise, puis termine par une ligne "Sources: ...".

        Question:
        {question}

        Extraits:
        {build_context(chunks)}
        """
    ).strip()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=450,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


def answer_with_openai(question: str, chunks: List[Chunk]) -> str:
    client = get_openai_client()
    if not client:
        raise RuntimeError("OPENAI_API_KEY absent.")
    prompt = textwrap.dedent(
        f"""
        Answer only from the provided excerpts.
        If the information is missing, say so clearly.
        Answer in French. Add a final line formatted exactly as "Sources: ...".

        Question:
        {question}

        Excerpts:
        {build_context(chunks)}
        """
    ).strip()
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return getattr(response, "output_text", "").strip()


def answer_extractively(chunks: List[Chunk]) -> str:
    best = chunks[0]
    return (
        "Aucun modele de generation n'est configure dans cette session. "
        "Voici l'extrait le plus pertinent trouve dans vos documents :\n\n"
        f"{best.text[:420].strip()}\n\nSources: {best.source_label}"
    )


def answer_question(question: str, chunks: List[Chunk]) -> str:
    if get_anthropic_client():
        return answer_with_anthropic(question, chunks)
    if get_openai_client():
        return answer_with_openai(question, chunks)
    return answer_extractively(chunks)


def process_uploaded_file(uploaded_file) -> None:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise RuntimeError(f"{uploaded_file.name} depasse {MAX_FILE_SIZE_MB} Mo.")
    file_path = save_uploaded_file(uploaded_file)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        extracted_chunks = extract_pdf_chunks(file_path)
        if not extracted_chunks:
            raise RuntimeError("Aucun texte exploitable trouve dans le PDF.")
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        image_text = extract_image_text(file_path)
        if not image_text:
            raise RuntimeError("Aucun texte detecte dans l'image.")
        extracted_chunks = [Chunk(file_path.name, file_path.name, piece) for piece in chunk_text(image_text)]
    elif suffix in {".mp3", ".wav", ".m4a", ".mp4"}:
        audio_text = extract_audio_text(file_path)
        if not audio_text:
            raise RuntimeError("Aucune transcription exploitable trouvee.")
        extracted_chunks = [Chunk(file_path.name, file_path.name, piece) for piece in chunk_text(audio_text)]
    else:
        raise RuntimeError("Format non supporte.")
    st.session_state.chunks.extend(extracted_chunks)
    st.session_state.documents.append({"name": file_path.name, "chunks": len(extracted_chunks)})


def render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp { background:
            radial-gradient(circle at top left, #f1ede2 0%, transparent 35%),
            linear-gradient(180deg, #fbfaf6 0%, #f2efe7 100%);
            color: #14211d;
        }
        .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 2rem; }
        .hero {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(20,33,29,0.1);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(20,33,29,0.06);
            margin-bottom: 1rem;
        }
        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #14211d;
            color: #f7f3ea;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
        }
        .hero h1 {
            font-size: clamp(2rem, 4vw, 3.4rem);
            margin: 0.7rem 0 0.2rem 0;
            line-height: 1;
        }
        .notice {
            background: #fff8e8;
            border-left: 4px solid #c88c1d;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            color: #5b4413;
        }
        .source-card {
            border: 1px solid rgba(20,33,29,0.08);
            border-radius: 18px;
            background: rgba(255,255,255,0.8);
            padding: 1rem;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="hero">
            <span class="eyebrow">DEMONSTRATEUR SOLVAI</span>
            <h1>{APP_TITLE}</h1>
            <p><strong>{APP_TAGLINE}</strong></p>
            <p>Testez notre IA sur vos propres documents. Aucun compte, aucune installation, aucune conservation apres la session.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SOLVAI Demo", page_icon="S", layout="centered", initial_sidebar_state="collapsed")
    ensure_session()
    render_header()

    with st.sidebar:
        st.write("Session temporaire")
        if st.button("Effacer cette session", use_container_width=True):
            reset_session()

    st.markdown(
        """
        <div class="notice">
            Ce systeme utilise l'intelligence artificielle. Les reponses sont generees automatiquement et supervisees par l'equipe SOLVAI.
            Les fichiers ne sont conserves que le temps de la session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Deposez vos fichiers ici",
        type=["pdf", "png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "mp4"],
        accept_multiple_files=True,
        help="PDF, image ou audio. 10 Mo maximum par fichier.",
    )

    if uploaded_files and st.button("Indexer les fichiers", type="primary", use_container_width=True):
        progress = st.progress(0, text="Indexation en cours...")
        errors = []
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                process_uploaded_file(uploaded_file)
            except Exception as exc:
                errors.append(f"{uploaded_file.name}: {exc}")
            progress.progress(idx / len(uploaded_files), text=f"Traitement de {uploaded_file.name}...")
        progress.empty()
        for error in errors:
            st.error(error)
        if st.session_state.documents:
            st.success("Indexation terminee.")

    if st.session_state.documents:
        st.markdown("### Fichiers indexes")
        for document in st.session_state.documents:
            st.markdown(f"- {document['name']} - {document['chunks']} extrait(s)")

    question = st.text_input(
        "Votre question",
        placeholder="Quelle est la procedure pour cloturer un dossier client ?",
    )

    if st.button("Obtenir la reponse", use_container_width=True):
        if not st.session_state.chunks:
            st.warning("Ajoutez et indexez au moins un fichier avant de poser une question.")
        elif not question.strip():
            st.warning("Posez une question.")
        else:
            with st.spinner("Recherche et generation de la reponse..."):
                top_chunks = retrieve_top_chunks(question.strip())
                if not top_chunks:
                    st.info("Aucun extrait pertinent n'a ete trouve.")
                else:
                    st.markdown("### Reponse")
                    st.write(answer_question(question.strip(), top_chunks))
                    st.markdown("### Sources")
                    for chunk in top_chunks:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>{chunk.source_label}</strong><br/>
                                {chunk.text[:420]}...
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
