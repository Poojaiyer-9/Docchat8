import streamlit as st
import os
import sys
import uuid

# Add project root to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ingest import ingest_pdf, clear_collection, get_indexed_docs
from app.rag import ask
from app.voice import text_to_speech
from app.llm_client import active_provider, is_ready, PROVIDER_INFO, DEFAULT_MODELS

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# --- Page config ---
st.set_page_config(
    page_title="DocChat - Local Voice RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- "Reading Room" theme: an ink-and-brass archival look, built to read
# like a document tool rather than another purple-gradient chat widget ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #14120e;
        --ink-2: #1c1912;
        --ink-3: #262119;
        --paper: #ece3ce;
        --paper-dim: #b7ab8f;
        --brass: #cc9a3c;
        --brass-bright: #e6b95c;
        --burgundy: #a34432;
        --forest: #5c8a63;
        --rule: rgba(204, 154, 60, 0.28);
    }

    * { box-sizing: border-box; }

    .stApp {
        background:
            radial-gradient(ellipse at 20% -10%, rgba(204,154,60,0.08) 0%, transparent 45%),
            var(--ink);
        font-family: 'Source Sans 3', sans-serif;
        color: var(--paper);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 { font-family: 'Fraunces', serif; }

    /* Header styling — a masthead, not a gradient hero */
    .header-container {
        background: var(--ink-2);
        padding: 2rem 2.5rem;
        border-radius: 6px;
        margin-bottom: 2rem;
        border: 1px solid var(--rule);
        border-top: 3px solid var(--brass);
        text-align: left;
        position: relative;
    }

    .header-container h1 {
        color: var(--paper);
        font-size: 2.75rem;
        font-weight: 700;
        font-style: italic;
        letter-spacing: -0.01em;
        margin-bottom: 0.35rem;
    }

    .header-container p {
        color: var(--paper-dim);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Sidebar cards */
    .sidebar-section {
        background: var(--ink-2);
        padding: 1.4rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        border: 1px solid var(--rule);
        border-left: 3px solid var(--brass);
        transition: border-color 0.2s ease;
    }

    .sidebar-section:hover {
        border-left-color: var(--brass-bright);
    }

    .sidebar-section h3 {
        color: var(--brass-bright);
        font-size: 1rem;
        font-weight: 600;
        font-style: italic;
        margin-bottom: 1rem;
    }

    /* Chat messages — margin-note cards, no filled bubbles */
    .stChatMessage {
        background: var(--ink-2);
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1.1rem;
        border: 1px solid var(--rule);
        border-left: 3px solid var(--paper-dim);
        animation: settleIn 0.25s ease;
    }

    @keyframes settleIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage[data-testid="user-message"] {
        border-left-color: var(--brass);
    }

    .stChatMessage[data-testid="assistant-message"] {
        border-left-color: var(--forest);
    }

    /* Citations — filed like index cards */
    .citation-box {
        background: var(--ink-3);
        border-left: 3px solid var(--brass);
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        transition: border-color 0.2s ease;
    }

    .citation-box:hover { border-left-color: var(--brass-bright); }

    .citation-box strong {
        color: var(--brass-bright);
        font-weight: 500;
    }

    /* Persona badge — a stamped tag, not a pill gradient */
    .persona-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        border: 1px solid currentColor;
    }

    .persona-student { color: var(--forest); background: rgba(92, 138, 99, 0.12); }
    .persona-researcher { color: var(--brass-bright); background: rgba(204, 154, 60, 0.12); }
    .persona-executive { color: var(--burgundy); background: rgba(163, 68, 50, 0.12); }

    audio {
        width: 100%;
        margin-top: 1rem;
        border-radius: 6px;
    }

    /* Loading animation */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--paper-dim);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        padding: 1rem;
    }

    .thinking-dots { display: flex; gap: 6px; }

    .thinking-dots span {
        width: 6px;
        height: 6px;
        background: var(--brass);
        border-radius: 1px;
        animation: bounce 1.4s infinite ease-in-out both;
    }

    .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.4); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Status indicator */
    .status-dot {
        height: 8px;
        width: 8px;
        background: var(--forest);
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }

    .status-dot.status-dot-warn { background: var(--burgundy); }

    /* Buttons */
    .stButton>button {
        background: transparent;
        color: var(--brass-bright);
        border: 1px solid var(--brass);
        border-radius: 4px;
        padding: 0.6rem 1.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: var(--brass);
        color: var(--ink);
    }

    .stSelectbox, .stFileUploader, .stTextInput {
        border-radius: 4px;
    }

    .stSuccess { border-left: 3px solid var(--forest); border-radius: 4px; }
    .stError { border-left: 3px solid var(--burgundy); border-radius: 4px; }
    .stInfo { border-left: 3px solid var(--brass); border-radius: 4px; }

    hr {
        border: none;
        height: 1px;
        background: var(--rule);
        margin: 2rem 0;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--ink); }
    ::-webkit-scrollbar-thumb { background: var(--brass); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--brass-bright); }
</style>
""", unsafe_allow_html=True)

# --- Session state initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Persona selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🎭 Persona")
    persona = st.selectbox(
        "Select answer style:",
        ["student", "researcher", "executive"],
        format_func=lambda x: {
            "student": "🎓 Student (Simple & Clear)",
            "researcher": "🔬 Researcher (Technical & Detailed)",
            "executive": "💼 Executive (Bullet Points)"
        }[x],
        help="Choose how you want answers formatted"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Document upload
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF to chat with it. The document will be processed locally."
    )
    
    if uploaded_file is not None:
        if st.session_state.current_doc != uploaded_file.name:
            with st.spinner("Processing document..."):
                # Save uploaded file
                temp_path = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # Clear previous collection and ingest new document
                    clear_collection()
                    num_chunks = ingest_pdf(temp_path, uploaded_file.name)
                    st.session_state.current_doc = uploaded_file.name
                    st.success(f"✅ Indexed {uploaded_file.name}\n📊 {num_chunks} chunks created")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
    else:
        st.session_state.current_doc = None
        st.info("👆 Upload a PDF to get started")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Document info
    if st.session_state.current_doc:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📚 Current Document")
        st.markdown(f"**{st.session_state.current_doc}**")
        docs = get_indexed_docs()
        if docs:
            st.caption(f"Sources: {', '.join(set(d['source'] for d in docs))}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System status — reflects whichever provider is actually configured,
    # instead of a hardcoded "Ollama: Running" that lies once this is deployed.
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🖥️ System Status")
    provider = active_provider()
    ready = is_ready()
    dot_class = "status-dot" if ready else "status-dot status-dot-warn"
    provider_label = PROVIDER_INFO.get(provider, {}).get("label", provider)
    status_text = "ready" if ready else "needs API key"
    st.markdown(f'<span class="{dot_class}"></span>**LLM:** {provider_label} — {status_text}',
                unsafe_allow_html=True)
    st.caption(f"Model: {DEFAULT_MODELS.get(provider, 'unknown')}")
    st.markdown('<span class="status-dot"></span>**Vector DB:** Chroma (local embeddings)',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main chat area ---
st.markdown("""
<div class="header-container">
    <h1>📄 DocChat</h1>
    <p>Your document assistant — local embeddings, bring your own LLM</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Show persona badge
            persona_class = f"persona-{message.get('persona', 'student')}"
            st.markdown(f'<span class="persona-badge {persona_class}">{message.get("persona", "student").title()}</span>', 
                       unsafe_allow_html=True)
        
        st.markdown(message["content"])
        
        # Show citations if available
        if "citations" in message and message["citations"]:
            with st.expander("📌 View Sources"):
                for citation in message["citations"]:
                    st.markdown(f"""
                    <div class="citation-box">
                        <strong>📄 {citation['source']}</strong><br>
                        📍 Page {citation['page']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show audio if available
        if "audio_url" in message and message["audio_url"]:
            st.audio(message["audio_url"], format="audio/mp3")

# --- Chat input ---
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.current_doc:
        st.error("Please upload a PDF document first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Show persona badge
            persona_class = f"persona-{persona}"
            st.markdown(f'<span class="persona-badge {persona_class}">{persona.title()}</span>', 
                       unsafe_allow_html=True)
            
            # Show thinking indicator
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                st.markdown("""
                <div class="thinking-indicator">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span>Thinking...</span>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                # Run RAG pipeline
                result = ask(prompt, persona, top_k=3)
                
                # Remove thinking indicator
                thinking_placeholder.empty()
                
                # Display answer
                st.markdown(result["answer"])
                
                # Display citations
                if result["citations"]:
                    with st.expander("📌 View Sources"):
                        for citation in result["citations"]:
                            st.markdown(f"""
                            <div class="citation-box">
                                <strong>📄 {citation['source']}</strong><br>
                                📍 Page {citation['page']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Generate and display audio
                audio_filename = f"answer_{uuid.uuid4().hex}.mp3"
                audio_path = text_to_speech(result["answer"], output_path=f"static/{audio_filename}")
                # Use absolute file path for st.audio (works in both local and Streamlit Cloud)
                abs_audio_path = os.path.abspath(audio_path)
                st.audio(abs_audio_path, format="audio/mp3")
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "audio_url": abs_audio_path,
                    "persona": persona
                })
                
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"❌ Error: {str(e)}")

# --- Voice Input Section ---
st.markdown("---")
st.markdown("### 🎤 Voice Input")

# Check if audio_input is available (Streamlit >= 1.28)
try:
    audio_input = st.audio_input("Record your question")
    if audio_input is not None:
        with st.spinner("Transcribing audio..."):
            try:
                from app.voice import speech_to_text
                # Save audio to temp file
                temp_audio = f"temp_voice_{uuid.uuid4().hex}.wav"
                with open(temp_audio, "wb") as f:
                    f.write(audio_input.getvalue())
                
                transcript = speech_to_text(temp_audio)
                
                if "Could not understand" in transcript or "unavailable" in transcript:
                    st.warning(f"⚠️ {transcript}. Please try again or use text input.")
                else:
                    st.success(f"🎤 Transcribed: **{transcript}**")
                    # Auto-send the transcribed question
                    if st.button("Send transcribed question", key="send_voice"):
                        st.session_state.messages.append({"role": "user", "content": transcript})
                        st.rerun()
            except Exception as e:
                st.warning(f"⚠️ Voice input error: {str(e)}. Please use text input instead.")
            finally:
                if os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except:
                        pass
except AttributeError:
    st.info("ℹ️ Voice input requires Streamlit 1.28+. Use the text input below to ask questions.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b7ab8f; font-size: 0.85rem; font-family: 'IBM Plex Mono', monospace; padding: 1rem;">
    <p>🔒 Local embeddings (Chroma) | 🔑 Bring your own LLM key | ⚡ FastAPI Backend</p>
    <p>Built with Streamlit | Your PDF never leaves this session's vector store</p>
</div>
""", unsafe_allow_html=True)
