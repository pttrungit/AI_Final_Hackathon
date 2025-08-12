"""
Healthcare AI Assistant with RAG + Medicine Image Recognition (FAISS version)

- Vector store: FAISS (local)
- Embeddings: HuggingFace all-MiniLM-L6-v2
- LLM: AzureChatOpenAI (preferred) or ChatOpenAI fallback
- Streamlit UI: Sidebar (about/features), Center chat + image upload, Right quick questions
- Session state + caching for heavy resources
"""

import os
import io
import base64
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load local .env for local dev
load_dotenv()

# ---------------- Robust imports (tolerant to langchain versions) ----------------
# Embeddings
_hf_ok = False
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    _hf_ok = True
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _hf_ok = True
    except Exception:
        _hf_ok = False

# FAISS vectorstore
_faiss_ok = False
try:
    from langchain.vectorstores import FAISS
    _faiss_ok = True
except Exception:
    try:
        from langchain_community.vectorstores import FAISS
        _faiss_ok = True
    except Exception:
        _faiss_ok = False

# LLM wrappers
_llm_wrapper = None
try:
    from langchain_openai import AzureChatOpenAI
    _llm_wrapper = "azure"
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI
        _llm_wrapper = "openai"
    except Exception:
        _llm_wrapper = None

# Core langchain pieces
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Validate imports
if not _hf_ok:
    st.error("HuggingFaceEmbeddings not found. Install 'sentence-transformers' and compatible langchain packages.")
    st.stop()
if not _faiss_ok:
    st.error("FAISS vectorstore not available. Install 'faiss-cpu' and compatible langchain packages.")
    st.stop()
if _llm_wrapper is None:
    st.warning("No Azure/OpenAI LLM wrapper found. LLM calls may fail if not configured.")


# ---------------- Config / Secrets helpers ----------------
def get_secret(key: str) -> Optional[str]:
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

# ---------------- Small Demo Medical KB (replace with full KB in prod) ----------------
MEDICAL_KNOWLEDGE_BASE = [
    {"title":"Common Cold","content":"The common cold is a viral infection of the upper respiratory tract. Symptoms include runny nose, sneezing, sore throat, cough, congestion, slight body aches, mild headache, low-grade fever. Treatment: rest, fluids, over-the-counter pain relievers. Prevention: hand washing, avoid touching face, avoid close contact with sick people."},
    {"title":"Aspirin","content":"Aspirin (acetylsalicylic acid) is a common pain reliever and anti-inflammatory. Uses: pain relief, fever reduction, anti-inflammatory, blood clot prevention. Dosage: 325-650mg every 4 hours for pain. Low dose (81mg) daily for heart disease prevention. Side effects: stomach upset, bleeding risk. Contraindications: bleeding disorders, children with fever."},
    {"title":"Ibuprofen","content":"Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). Uses: pain relief, fever reduction, inflammation reduction. Dosage: 200-400mg every 4-6 hours, maximum 1200mg/day without prescription. Side effects: stomach upset, dizziness, rash."},
    {"title":"Paracetamol","content":"Paracetamol (acetaminophen) is a pain reliever and fever reducer. Dosage: 500-1000mg every 4-6 hours, max 4000mg/day for adults. Overdose causes liver damage."},
    {"title":"Metformin","content":"Metformin is first-line treatment for type 2 diabetes. Start 500mg once or twice daily; may be increased up to 2000mg/day. Side effects: nausea, diarrhea."}
]

# ---------------- Helpers ----------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def safe_get(obj, attr, default=None):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

# ---------------- Cached heavy resources ----------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def build_faiss_index(persist_dir: str = "./faiss_index"):
    """
    Build FAISS vectorstore from the in-memory MEDICAL_KNOWLEDGE_BASE.
    Saves locally to persist_dir.
    """
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for item in MEDICAL_KNOWLEDGE_BASE:
        chunks = splitter.split_text(item["content"])
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"title": item["title"], "source":"Medical KB"}))
    # Create FAISS index
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    # Try to persist to disk
    try:
        # Many langchain FAISS implementations offer save_local
        vectorstore.save_local(persist_dir)
    except Exception:
        try:
            vectorstore.save_local(persist_dir)  # attempt again or other variant
        except Exception:
            # Not fatal: index stays in memory for session
            st.info("FAISS index built in-memory (persistence not supported in this environment).")
    return vectorstore

@st.cache_resource
def load_faiss_index(persist_dir: str = "./faiss_index"):
    """
    Load FAISS index if persisted; if not, build a new one.
    """
    embeddings = get_embeddings()
    # Try load_local
    try:
        vs = FAISS.load_local(persist_dir, embeddings)
        return vs
    except Exception:
        # Build new
        return build_faiss_index(persist_dir=persist_dir)

@st.cache_resource
def build_llm():
    """
    Return LLM instance. Prefer AzureChatOpenAI (if configured), otherwise ChatOpenAI if OPENAI_API_KEY present.
    """
    # Try Azure
    if _llm_wrapper == "azure" and AZURE_OPENAI_API_KEY and AZURE_DEPLOYMENT_NAME:
        os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        if AZURE_OPENAI_ENDPOINT:
            os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
        try:
            llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_VERSION,
                temperature=0.2,
                max_tokens=600
            )
            return llm
        except Exception as e:
            st.warning(f"AzureChatOpenAI init failed: {e}")

    # Try ChatOpenAI fallback
    if _llm_wrapper == "openai" and OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        try:
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini", max_tokens=600, openai_api_key=OPENAI_API_KEY)
            return llm
        except Exception as e:
            st.warning(f"ChatOpenAI init failed: {e}")

    return None

# ---------------- Image processing ----------------
def preprocess_image(image: Image.Image, max_dim: int = 1200) -> bytes:
    """Normalize PIL image to RGB JPEG bytes and optionally resize."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255,255,255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode not in ("RGB","L"):
        image = image.convert("RGB")
    # Resize if large
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize((int(image.width*ratio), int(image.height*ratio)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

# ---------------- Vision via LLM (attempt) ----------------
def analyze_image_with_llm(llm, img_bytes: bytes) -> str:
    """Attempt to call LLM with image content. Returns text analysis or error message."""
    if llm is None:
        return "Vision/LLM not configured. Please set Azure/OpenAI keys and ensure langchain wrappers are installed."
    b64 = image_to_base64(img_bytes)
    # Build messages; prefer SystemMessage/HumanMessage if available
    try:
        from langchain.schema.messages import SystemMessage, HumanMessage
        sys_msg = SystemMessage(content=(
            "You are a medication identification expert. Analyze the medicine image and identify:\n"
            "1) Likely medication name (generic and brand if visible)\n"
            "2) Dosage if visible\n"
            "3) Form (tablet, capsule, etc.)\n"
            "4) Color and shape\n"
            "5) Any visible markings or imprints\n"
            "If you are not certain, say so. Always advise verification with a pharmacist."
        ))
        human_msg = HumanMessage(content=[
            {"type":"text","text":"Identify this medication from the image and list possible matches."},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        ])
        # Try .invoke
        if hasattr(llm, "invoke"):
            resp = llm.invoke([sys_msg, human_msg])
            return safe_get(resp, "content", str(resp))
        # Try direct call
        if callable(llm):
            prompt = sys_msg.content + "\n\nImage (base64, truncated):\n" + b64[:500] + "...[truncated]\n\nPlease analyze."
            out = llm(prompt)
            return str(out)
    except Exception as e:
        # fallback: send textual prompt with note that image is base64
        try:
            prompt = (
                "You are a medication expert. The user provides an image encoded in base64 (truncated). "
                "Describe likely medication and markings if possible.\n\n"
                f"Image (base64, truncated): {b64[:500]}...[truncated]"
            )
            if hasattr(llm, "invoke"):
                resp = llm.invoke([{"role":"system","content":"You are a medication expert."},{"role":"user","content":prompt}])
                return safe_get(resp, "content", str(resp))
            elif callable(llm):
                return str(llm(prompt))
        except Exception as ex:
            return f"Vision analysis failed: {e}; fallback failed: {ex}"
    return "Vision analysis not available for this LLM wrapper."

# ---------------- RAG chain builder ----------------
@st.cache_resource
def build_rag_chain(faiss_dir: str = "./faiss_index"):
    """
    Build ConversationalRetrievalChain backed by FAISS
    """
    vectorstore = load_faiss_index(faiss_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
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
            combine_docs_chain_kwargs={"prompt":PROMPT},
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.warning(f"Could not initialize ConversationalRetrievalChain: {e}")
        return None

# ---------------- fallback/analysis helpers ----------------
NO_INFO_INDICATORS = ["i don't know","no information","not found","cannot find","i'm not sure","don't have information"]

def answer_indicates_no_info(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    return any(ind in t for ind in NO_INFO_INDICATORS)

def call_llm_direct(llm, query: str) -> str:
    if llm is None:
        return "LLM not configured. Please set Azure/OpenAI keys in secrets or environment."
    prompt = (
        "You are a medical information assistant. Provide accurate, evidence-based information. "
        "If uncertain, say so and advise consulting a healthcare professional.\n\n"
        f"User query: {query}\n\nAnswer with clear sections (Definition, Symptoms, Treatment, When to seek help)."
    )
    try:
        if hasattr(llm, "invoke"):
            resp = llm.invoke([{"role":"system","content":"You are a medical assistant."},{"role":"user","content":prompt}])
            return safe_get(resp, "content", str(resp))
        elif callable(llm):
            out = llm(prompt)
            return str(out)
    except Exception as e:
        return f"LLM call failed: {e}"

def add_medical_disclaimer(text: str) -> str:
    return text + "\n\n⚠️ **Medical Disclaimer**: For educational purposes only. Consult a healthcare professional for medical advice. For medication identification, verify with a licensed pharmacist."

# ---------------- Session state initialization ----------------
def initialize_session_state():
    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = load_faiss_index()
        except Exception as e:
            st.error(f"Error initializing FAISS vectorstore: {e}")
            st.stop()
    if "llm" not in st.session_state:
        st.session_state.llm = build_llm()
    if "assistant_chain" not in st.session_state:
        st.session_state.assistant_chain = build_rag_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "upload_preview" not in st.session_state:
        st.session_state.upload_preview = None

# ---------------- UI helpers ----------------
def add_user_message(content: str, image: Optional[Image.Image] = None):
    msg = {"role":"user","content":content,"time":now_iso()}
    if image is not None:
        msg["image"] = True
    st.session_state.chat_history.append(msg)

def add_assistant_message(content: str, sources: Optional[List[str]] = None):
    msg = {"role":"assistant","content":content,"time":now_iso(),"sources": sources or []}
    st.session_state.chat_history.append(msg)

def display_chat_history():
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            with st.chat_message("user"):
                if m.get("image") and st.session_state.get("upload_preview"):
                    st.image(st.session_state["upload_preview"], width=220)
                st.write(m["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(m["content"])
                if m.get("sources"):
                    with st.expander("📚 Sources"):
                        for s in m["sources"]:
                            st.write(f"- {s}")

# ---------------- Quick questions and medicines ----------------
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
COMMON_MEDICINES = ["Aspirin","Ibuprofen","Paracetamol (Acetaminophen)","Metformin","Amoxicillin","Omeprazole"]

# ---------------- Core processing ----------------
def process_query(query: str):
    chain = st.session_state.get("assistant_chain")
    llm = st.session_state.get("llm")

    # Prefer RAG chain if available
    if chain:
        try:
            res = chain({"question": query})
            answer = res.get("answer", "")
            if answer_indicates_no_info(answer):
                # fallback to LLM
                fallback = call_llm_direct(llm, query)
                combined = f"**(Not found in KB — fallback to LLM)**\n\n{fallback}"
                add_assistant_message(add_medical_disclaimer(combined), sources=[])
            else:
                docs = res.get("source_documents", [])
                sources = []
                for d in docs:
                    try:
                        sources.append(d.metadata.get("title","Unknown"))
                    except Exception:
                        pass
                add_assistant_message(add_medical_disclaimer(answer), sources=list(dict.fromkeys(sources)))
            return
        except Exception as e:
            st.warning(f"RAG chain error: {e}. Falling back to direct LLM.")

    # If no chain or error, use direct LLM
    out = call_llm_direct(llm, query)
    add_assistant_message(add_medical_disclaimer(out), sources=[])

# ---------------- FAISS persistence helpers ----------------
def load_faiss_index(persist_dir: str = "./faiss_index"):
    """
    Try to load FAISS index if present; otherwise build.
    """
    embeddings = get_embeddings()
    # Try load_local
    try:
        vs = FAISS.load_local(persist_dir, embeddings)
        return vs
    except Exception:
        # build new
        return build_faiss_index(persist_dir)

# ---------------- Main Streamlit app UI ----------------
def main():
    st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")
    initialize_session_state()

    st.markdown("<h1 style='font-size:34px;'>🏥 Healthcare AI Assistant with Medicine Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Your intelligent medical guidance companion powered by RAG and image analysis.")

    left_col, main_col, right_col = st.columns([1.2, 3, 1.0])

    # --- Left Sidebar ---
    with left_col:
        st.markdown("### ℹ️ About")
        st.info("Medical information, symptom checking, medication guidance, and pill identification from images.")
        st.markdown("### 🔧 Features")
        st.markdown("- 📷 Medicine Image Analysis\n- 🔍 Symptom Analysis\n- 💊 Medication Info\n- 🩺 Disease Info & Health Tips")
        st.markdown("### ⚠️ Important")
        st.warning("This assistant provides general information only. Always verify medication identification with a pharmacist and consult healthcare professionals.")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            try:
                if st.session_state.assistant_chain and hasattr(st.session_state.assistant_chain,"memory"):
                    st.session_state.assistant_chain.memory.clear()
            except Exception:
                pass
            st.rerun()

    # --- Right quick questions ---
    with right_col:
        st.markdown("### 📋 Quick Questions")
        for q in QUICK_QUESTIONS:
            if st.button(q, key=f"quick_{q}"):
                add_user_message(q)
                process_query(q)
                st.rerun()
        st.markdown("### 💊 Common Medicines")
        st.info("\n".join(f"- {m}" for m in COMMON_MEDICINES))

    # --- Main center: upload + chat ---
    with main_col:
        st.markdown("## 💬 Chat with Healthcare Assistant")

        # Image upload
        with st.expander("📷 Upload Medicine Image for Identification", expanded=True):
            uploaded_file = st.file_uploader("Upload clear image of the medicine (pill/tablet/capsule). Show imprints if possible.", type=['png','jpg','jpeg'])
            if uploaded_file:
                try:
                    pil_img = Image.open(uploaded_file)
                    st.session_state.upload_preview = pil_img
                    st.image(pil_img, caption="Uploaded Medicine", width=320)
                except Exception as e:
                    st.error(f"Unable to read image: {e}")

                if st.button("🔍 Analyze Medicine"):
                    try:
                        img_bytes = preprocess_image(pil_img)
                        add_user_message("Please identify this medicine from the image.", image=pil_img)
                        with st.spinner("Analyzing image..."):
                            analysis = analyze_image_with_llm(st.session_state.llm, img_bytes)
                            # Try RAG search for details
                            rag = st.session_state.assistant_chain
                            additional = None
                            sources = []
                            if rag:
                                try:
                                    search_q = f"Provide details about this medication candidate: {analysis[:200]}"
                                    rag_res = rag({"question": search_q})
                                    additional = rag_res.get("answer", None)
                                    docs = rag_res.get("source_documents", [])
                                    for d in docs:
                                        try:
                                            sources.append(d.metadata.get("title","Unknown"))
                                        except Exception:
                                            pass
                                except Exception:
                                    additional = None
                            text = f"**Image Analysis Results:**\n\n{analysis}"
                            if additional:
                                text += f"\n\n**Additional info (KB):**\n{additional}"
                            # add disclaimer, sources
                            add_assistant_message(add_medical_disclaimer(text), sources=list(dict.fromkeys(sources)))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Image analysis failed: {e}\n{traceback.format_exc()}")

        # Show chat history
        display_chat_history()

        # Chat input
        user_input = st.chat_input("Ask about symptoms, medications, or health conditions...")
        if user_input:
            add_user_message(user_input)
            process_query(user_input)
            st.rerun()

# ---------------- Start app ----------------
if __name__ == "__main__":
    main()