"""
Healthcare AI Assistant with RAG + Medicine Image Recognition
Full-featured Streamlit app (single-file) implementing:
- UI: Sidebar (About, Features, Important), Center Chat w/ image upload, Right Quick Questions & Common Medicines
- RAG: Chroma vectorstore + HuggingFace embeddings, conversational retrieval
- Vision: Vision-capable LLM (AzureChatOpenAI) used for image analysis (placeholder fallback available)
- Query router, Image processor, Error handler, Response handler
- Uses streamlit session_state, caches heavy resources with @st.cache_resource

Before running:
- Set secrets in Streamlit Cloud or local env:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT (optional)
    - AZURE_DEPLOYMENT_NAME
    - AZURE_OPENAI_VERSION (optional)
- Or set OPENAI_API_KEY if using ChatOpenAI fallback.

requirements (summary):
    streamlit>=1.20.0
    pillow
    python-dotenv
    langchain>=0.0.x
    langchain-openai
    langchain-community or chromadb (depending on langchain version)
    chromadb
    sentence-transformers
"""

import os
import io
import json
import base64
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# load local .env if present (for local dev)
load_dotenv()

# ---- Robust imports for LangChain / Vectorstores / LLM wrappers ----
# Try a few variants to be tolerant to version differences
_chroma_ok = False
_hf_ok = False
_llm_wrapper = None
try:
    # embeddings
    from langchain.embeddings import HuggingFaceEmbeddings
    _hf_ok = True
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _hf_ok = True
    except Exception:
        _hf_ok = False

# Chroma vectorstore import (varies by version)
try:
    from langchain.vectorstores import Chroma
    _chroma_ok = True
except Exception:
    try:
        # older/newer combos
        from langchain_community.vectorstores import Chroma
        _chroma_ok = True
    except Exception:
        try:
            import chromadb
            from langchain.vectorstores import Chroma
            _chroma_ok = True
        except Exception:
            _chroma_ok = False

# LLM wrappers
_azure_ok = False
_openai_ok = False
try:
    from langchain_openai import AzureChatOpenAI
    _azure_ok = True
    _llm_wrapper = "azure"
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI
        _openai_ok = True
        _llm_wrapper = "openai"
    except Exception:
        _llm_wrapper = None

# Core langchain building blocks (prompts, chains, memory)
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---- Config / Secrets ----
# Use st.secrets for Streamlit Cloud. For local dev, rely on environment variables or .env.
def get_secret(key: str) -> Optional[str]:
    # prefer st.secrets if available
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = get_secret("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = get_secret("AZURE_OPENAI_VERSION") or "2024-07-01-preview"
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# Validate minimal config
if not _hf_ok:
    st.error("HuggingFaceEmbeddings not available. Install 'sentence-transformers' and 'langchain-community' or appropriate langchain package.")
    st.stop()
if not _chroma_ok:
    st.error("Chroma vectorstore not available. Install 'chromadb' and compatible 'langchain' package.")
    st.stop()
if _llm_wrapper is None:
    st.warning("No Azure or ChatOpenAI wrapper found. The app will still run but LLM calls may fail. Install 'langchain-openai' or update 'langchain' to include chat_models.")
    # Do not stop here; allow placeholder/fallback

# ---- Medical Knowledge Base (sample, can be expanded / loaded from files) ----
MEDICAL_KNOWLEDGE_BASE = [
    {"title":"Common Cold","content":"The common cold is a viral infection... (see full KB in production)"},
    {"title":"Aspirin","content":"Aspirin (acetylsalicylic acid) is a common pain reliever..."},
    {"title":"Ibuprofen","content":"Ibuprofen is an NSAID..."},
    {"title":"Paracetamol","content":"Paracetamol (acetaminophen) is a pain reliever..."},
    {"title":"Metformin","content":"Metformin used in Type 2 diabetes..."},
    # You may expand with the full KB entries you provided earlier
]

# ---- Utility helpers ----
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ---- Caching heavy resources ----
@st.cache_resource
def get_embeddings():
    # CPU-friendly huggingface embeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def get_chroma(persist_dir: str = "./chroma_db"):
    # Create or load Chroma vectorstore
    embeddings = get_embeddings()
    # Chroma.from_documents will create and persist; allow reloading
    try:
        # create documents from KB if not exists
        # We build doc list each time to be safe
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for item in MEDICAL_KNOWLEDGE_BASE:
            chunks = splitter.split_text(item["content"])
            for c in chunks:
                docs.append(Document(page_content=c, metadata={"title": item["title"], "source":"Medical KB"}))
        vect = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
        try:
            vect.persist()
        except Exception:
            # Some Chroma wrappers automatically persist; ignore if not supported
            pass
        return vect
    except Exception as e:
        st.error(f"Error initializing Chroma vectorstore: {e}")
        raise

@st.cache_resource
def build_llm():
    # Prefer AzureChatOpenAI; fallback to ChatOpenAI
    if _llm_wrapper == "azure" and AZURE_OPENAI_API_KEY and AZURE_DEPLOYMENT_NAME:
        # set environment variables used by some SDKs
        os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        if AZURE_OPENAI_ENDPOINT:
            os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
        try:
            llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_VERSION,
                temperature=0.2,
                max_tokens=600,
            )
            return llm
        except Exception as e:
            st.warning(f"AzureChatOpenAI instantiation failed: {e}. Trying ChatOpenAI fallback.")

    if _llm_wrapper == "openai" and OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        try:
            llm = None
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini", max_tokens=600, openai_api_key=OPENAI_API_KEY)
            return llm
        except Exception as e:
            st.warning(f"ChatOpenAI instantiation failed: {e}")

    # If none available, return None; app will use placeholder responses or error messages
    return None

# ---- Image processor ----
def preprocess_image(image: Image.Image) -> bytes:
    """
    Convert uploaded PIL image to a standard JPEG RGB bytes (suitable for base64 embedding)
    """
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255,255,255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    # Resize if extremely large to avoid huge payloads (maintain aspect)
    max_dim = 1200
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize((int(image.width*ratio), int(image.height*ratio)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

# ---- Vision integration (via LLM chat if it supports images) ----
def analyze_image_with_llm(llm, img_bytes: bytes) -> str:
    """
    Try to call the LLM/vision model to analyze medicine image.
    This function attempts a few message shapes depending on what the LLM supports.
    Returns textual analysis string.
    """
    if llm is None:
        return "Vision model not configured. Install and configure AzureChatOpenAI or ChatOpenAI with vision support."

    base64_image = image_to_base64(img_bytes)

    # Preferred: if llm has 'invoke' method, try sending SystemMessage / HumanMessage
    try:
        from langchain.schema.messages import SystemMessage, HumanMessage
        system = SystemMessage(content=(
            "You are a medication identification expert. Analyze the medicine image and identify:\n"
            "1) Likely medication name (generic and brand if visible)\n"
            "2) Dosage if visible\n"
            "3) Form (tablet, capsule, etc.)\n"
            "4) Color and shape\n"
            "5) Any visible markings or imprints\n"
            "If you are not certain, say so. Always advise verification with a pharmacist."
        ))
        human_content = [
            {"type": "text", "text": "Please identify the medication in the image. Describe what you see and list possible matches."},
            {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        human = HumanMessage(content=human_content)
        # Some wrappers expect a list of messages or a single call to invoke
        if hasattr(llm, "invoke"):
            resp = llm.invoke([system, human])
            # resp.content is typical
            return safe_get(resp.__dict__, "content") or str(resp)
        elif hasattr(llm, "chat"):
            # try generic chat
            resp = llm.chat([system, human])
            # resp may have .content or .generations
            text = safe_get(resp.__dict__, "content")
            if text:
                return text
            return str(resp)
        elif hasattr(llm, "__call__"):
            # fallback: send text-only prompt describing that image is base64
            prompt = (
                    system.content + "\n\n"
                                     "Image (base64):\n"
                                     f"{base64_image[:500]}...[truncated]\n\n"
                                     "Please analyze and respond."
            )
            out = llm(prompt)
            return str(out)
    except Exception as e:
        # Last fallback: provide simple guidance
        return f"Vision call failed: {e}\n(If you configured Azure/OpenAI with vision-capable model, ensure wrapper supports image messages.)"

# ---- RAG / QA chain builder ----
@st.cache_resource
def build_rag_chain(chroma_dir: str = "./chroma_db"):
    chroma = get_chroma(persist_dir=chroma_dir)
    retriever = chroma.as_retriever(search_kwargs={"k": 3})
    llm = build_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt_template = (
        "You are a professional healthcare AI assistant. Use the context below to answer the user's question.\n"
        "Always remind the user to seek professional medical advice when necessary.\n\n"
        "Context: {context}\n\n"
        "Chat History: {chat_history}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    )
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","chat_history","question"])
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.warning(f"Could not build ConversationalRetrievalChain automatically: {e}. You may still use direct LLM fallback.")
        return None

# ---- Response & Error handling ----
NO_INFO_INDICATORS = [
    "i don't know", "i do not know", "no information", "not in my knowledge",
    "cannot find", "no relevant", "i'm not sure", "not found"
]

def analyze_chain_answer_for_missing(answer: str) -> bool:
    if not answer:
        return True
    al = answer.lower()
    for ind in NO_INFO_INDICATORS:
        if ind in al:
            return True
    return False

# ---- Initialize session_state ----
def initialize_session_state():
    if "assistant_chain" not in st.session_state:
        st.session_state.assistant_chain = build_rag_chain()
    if "llm" not in st.session_state:
        st.session_state.llm = build_llm()
    if "chroma" not in st.session_state:
        try:
            st.session_state.chroma = get_chroma()
        except Exception:
            st.session_state.chroma = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_user_message" not in st.session_state:
        st.session_state.last_user_message = ""
    if "upload_preview" not in st.session_state:
        st.session_state.upload_preview = None

# ---- UI helpers ----
def add_user_message(content: str, image: Optional[Image.Image] = None):
    entry = {"role":"user","content":content,"time": now_iso()}
    if image is not None:
        entry["image"] = True
    st.session_state.chat_history.append(entry)

def add_assistant_message(content: str, sources: Optional[List[str]] = None):
    entry = {"role":"assistant","content":content,"time": now_iso(), "sources": sources or []}
    st.session_state.chat_history.append(entry)

def display_chat_history():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                if message.get("image") and st.session_state.get("upload_preview"):
                    st.image(st.session_state["upload_preview"], width=220)
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for s in message["sources"]:
                            st.write(f"- {s}")

# ---- Quick questions and common medicines (static lists) ----
QUICK_QUESTIONS = [
    "What are the symptoms of flu?",
    "Tell me about aspirin dosage",
    "How to manage diabetes?",
    "What causes headaches?",
    "COVID-19 prevention tips",
    "Allergy treatment options",
    "High blood pressure info",
    "Paracetamol vs Ibuprofen?"
]

COMMON_MEDICINES = [
    "Aspirin", "Ibuprofen", "Paracetamol (Acetaminophen)", "Metformin",
    "Amoxicillin", "Omeprazole"
]

# ---- Main App UI ----
def main():
    st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")
    initialize_session_state()

    # Top header
    st.markdown(
        "<h1 style='font-size:36px;'>🏥 Healthcare AI Assistant with Medicine Recognition</h1>",
        unsafe_allow_html=True)
    st.markdown("Your intelligent medical guidance companion powered by RAG technology and image analysis")

    # layout: left sidebar, main, right quick
    left_col, main_col, right_col = st.columns([1.2, 3, 1.1])

    # --- LEFT SIDEBAR ---
    with left_col:
        st.markdown("### ℹ️ About")
        st.info("This AI assistant provides medical information, symptom checking, medication guidance, and can identify medicines from images using RAG and vision AI.")
        st.markdown("### 🔧 Features")
        st.markdown("""
        - 📷 Medicine Image Analysis (upload clear pill photos)
        - 🔍 Symptom Analysis
        - 💊 Medication Info & Dosage
        - 🩺 Disease Information & Health Tips
        """)
        st.markdown("### ⚠️ Important")
        st.warning("This assistant provides general information only. Always verify medication identification with a pharmacist. Consult healthcare professionals for medical decisions.")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            # clear memory object if present
            try:
                if st.session_state.assistant_chain and hasattr(st.session_state.assistant_chain, "memory"):
                    st.session_state.assistant_chain.memory.clear()
            except Exception:
                pass
            st.rerun()

    # --- RIGHT quick questions ---
    with right_col:
        st.markdown("### 📋 Quick Questions")
        for q in QUICK_QUESTIONS:
            if st.button(q, key=f"quick_{q}"):
                # push into flow
                st.session_state.last_user_message = q
                add_user_message(q)
                process_user_query(q)
                st.rerun()

        st.markdown("### 💊 Common Medicines")
        st.info("\n".join(f"- {m}" for m in COMMON_MEDICINES))

    # --- MAIN CENTER: Chat and Upload ---
    with main_col:
        st.markdown("## 💬 Chat with Healthcare Assistant")

        # Image upload expander
        with st.expander("📷 Upload Medicine Image for Identification", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload a clear image of the medicine (pill, tablet, capsule). Show any imprints.",
                type=['png','jpg','jpeg'], accept_multiple_files=False
            )
            if uploaded_file:
                try:
                    pil_image = Image.open(uploaded_file)
                    st.session_state["upload_preview"] = pil_image
                    st.image(pil_image, caption="Uploaded Medicine", width=320)
                except Exception as e:
                    st.error(f"Could not read image: {e}")

                if st.button("🔍 Analyze Medicine"):
                    # Preprocess and call vision
                    try:
                        img_bytes = preprocess_image(pil_image)
                        add_user_message("Please identify this medicine from the image.", image=pil_image)
                        with st.spinner("Analyzing medicine image..."):
                            llm = st.session_state.llm
                            analysis = analyze_image_with_llm(llm, img_bytes)
                            # after analysis, do a KB search for additional info
                            rag = st.session_state.assistant_chain
                            detailed_info = None
                            sources = []
                            if rag:
                                # Form a query based on analysis summary (short)
                                search_q = f"Provide details about this medication candidate: {analysis[:200]}"
                                try:
                                    rag_result = rag({"question": search_q})
                                    detailed_info = safe_get(rag_result, "answer", default=None) or None
                                    docs = safe_get(rag_result, "source_documents", default=[])
                                    if docs:
                                        for d in docs:
                                            try:
                                                sources.append(d.metadata.get("title", "Unknown"))
                                            except Exception:
                                                pass
                                except Exception as e:
                                    # ignore rag failure; we'll still return analysis
                                    detailed_info = None
                            # Build assistant response
                            response_text = f"**Image Analysis Results:**\n\n{analysis}"
                            if detailed_info:
                                response_text += f"\n\n**Additional Info (from KB):**\n{detailed_info}"
                            # Append disclaimer
                            response_text = add_medical_disclaimer(response_text)
                            add_assistant_message(response_text, sources=list(dict.fromkeys(sources)))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Image analysis failed: {e}\n{traceback.format_exc()}")

        # Display chat history
        display_chat_history()

        # Chat input
        user_input = st.chat_input("Ask about symptoms, medications, or health conditions...")
        if user_input:
            st.session_state.last_user_message = user_input
            add_user_message(user_input)
            process_user_query(user_input)
            st.rerun()

# ---- Core processing pipeline (query router + handlers) ----
def process_user_query(query: str):
    """
    Decide whether to route to RAG or direct LLM, run the chain or LLM, handle fallback,
    and append an assistant response into chat history.
    """
    rag = st.session_state.get("assistant_chain")
    llm = st.session_state.get("llm")

    # Try RAG first if available
    if rag:
        try:
            with st.spinner("Retrieving from knowledge base..."):
                result = rag({"question": query})
            answer = result.get("answer", "")
            # If RAG didn't find good info, fallback to GPT
            if analyze_chain_answer_for_missing(answer):
                # fallback to LLM
                fallback_text = call_llm_direct(llm, query)
                combined = f"**(Fallback from KB)**\n\n{fallback_text}"
                combined = add_medical_disclaimer(combined)
                add_assistant_message(combined, sources=[])
            else:
                # Good answer from KB
                answer_with_disclaimer = add_medical_disclaimer(answer)
                docs = result.get("source_documents", [])
                source_titles = []
                for d in docs:
                    try:
                        source_titles.append(d.metadata.get("title","Unknown"))
                    except Exception:
                        pass
                add_assistant_message(answer_with_disclaimer, sources=list(dict.fromkeys(source_titles)))
            return
        except Exception as e:
            # On chain error, fallback to direct LLM
            st.warning(f"RAG chain error: {e}. Falling back to direct LLM.")
    # If no rag or failed, use LLM
    fallback_text = call_llm_direct(llm, query)
    fallback_text = add_medical_disclaimer(fallback_text)
    add_assistant_message(fallback_text, sources=[])

def call_llm_direct(llm, query: str) -> str:
    """
    Call LLM directly to produce medical information. Returns string.
    """
    if llm is None:
        return "LLM not configured. Please set AZURE_OPENAI_API_KEY (Azure) or OPENAI_API_KEY (OpenAI) and ensure langchain wrappers are installed."

    # Build a safety prompt structure for medical info
    prompt = (
        "You are a medical information assistant. Provide accurate, evidence-based information. "
        "If uncertain, say so and advise consulting a healthcare professional.\n\n"
        f"User Query: {query}\n\n"
        "Answer in a clear, structured manner (Definition, Symptoms, Treatment, When to seek help)."
    )
    try:
        # Try common interfaces
        if hasattr(llm, "invoke"):
            messages = [
                {"role":"system","content":"You are a helpful medical assistant."},
                {"role":"user","content": prompt}
            ]
            resp = llm.invoke(messages)
            return safe_get(resp.__dict__, "content") or str(resp)
        elif hasattr(llm, "generate"):
            # some wrappers use generate with messages or prompt
            resp = llm.generate([{"role":"user","content": prompt}])
            # extract text if possible
            try:
                gens = resp.generations
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    return getattr(gens[0][0], "text", str(gens[0][0]))
            except Exception:
                return str(resp)
        elif callable(llm):
            out = llm(prompt)
            return str(out)
    except Exception as e:
        return f"LLM call error: {e}"

def add_medical_disclaimer(response: str) -> str:
    return response + "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice. For medication identification, always verify with a licensed pharmacist."

# ---- Run app ----
if __name__ == "__main__":
    main()
