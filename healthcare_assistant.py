"""
Healthcare AI Assistant with RAG and Medicine Image Recognition
Streamlit Cloud Compatible Version

This version fixes imports and dependency issues for Streamlit Cloud deployment.
"""

import streamlit as st
import os
from typing import List, Dict, Any
import base64
from datetime import datetime
from PIL import Image
import io
import traceback

# === Robust imports for LangChain ecosystem ===
# Prefer the community package where HuggingFace embeddings and FAISS live now.
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    langchain_community_available = True
except Exception:
    # Fallback to older langchain names (some environments still have these)
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        langchain_community_available = False
    except Exception:
        st.error(
            "Required LangChain embedding/vectorstore packages are not installed.\n"
            "Please add `langchain-community` or `langchain` + `faiss-cpu` to requirements.txt and redeploy."
        )
        st.stop()

# Core langchain pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Azure / OpenAI LLM wrapper
# Try to import AzureChatOpenAI first, otherwise fallback to langchain.chat_models.ChatOpenAI
llm_class = None
try:
    from langchain_openai import AzureChatOpenAI
    llm_class = "azure"
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI
        llm_class = "openai"
    except Exception:
        st.error(
            "No compatible Chat LLM class found. Please ensure you installed `langchain-openai` "
            "or `langchain` with chat model support."
        )
        st.stop()

# === Azure OpenAI Configuration with Streamlit Secrets / env ===
AZURE_OPENAI_API_KEY = None
AZURE_OPENAI_ENDPOINT = None
AZURE_DEPLOYMENT_NAME = None
AZURE_OPENAI_VERSION = None

# Load from Streamlit secrets if available (Streamlit Cloud)
if isinstance(st.secrets, dict) and "AZURE_OPENAI_API_KEY" in st.secrets:
    AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT_NAME = st.secrets.get("AZURE_DEPLOYMENT_NAME")
    AZURE_OPENAI_VERSION = st.secrets.get("AZURE_OPENAI_VERSION")
else:
    # Local dev fallback — expect user to set environment variables.
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
    AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")

# === Basic Medical Knowledge Base (small for demo) ===
MEDICAL_KNOWLEDGE_BASE = [
    {
        "title": "Common Cold",
        "content": (
            "The common cold is a viral infection of the upper respiratory tract. "
            "Symptoms include: runny nose, sneezing, sore throat, cough, congestion, slight body aches, "
            "mild headache, low-grade fever. Treatment: rest, fluids, over-the-counter pain relievers. "
            "Prevention: hand washing, avoid touching face, avoid close contact with sick people."
        )
    },
    {
        "title": "Aspirin",
        "content": (
            "Aspirin (acetylsalicylic acid) is a common pain reliever and anti-inflammatory. "
            "Uses: pain relief, fever reduction, anti-inflammatory, blood clot prevention. "
            "Dosage: 325-650mg every 4 hours for pain. Low dose (81mg) daily for heart disease prevention. "
            "Side effects: stomach upset, bleeding risk. Contraindications: bleeding disorders, children with fever."
        )
    },
    # Add other entries as needed...
]

# === Helper: Safe LLM constructor ===
def build_llm():
    """Construct LLM instance based on available wrappers and configuration."""
    try:
        if llm_class == "azure":
            # Require azure config
            if not AZURE_OPENAI_API_KEY or not AZURE_DEPLOYMENT_NAME:
                st.error(
                    "Azure OpenAI credentials or deployment name missing. "
                    "Set AZURE_OPENAI_API_KEY and AZURE_DEPLOYMENT_NAME in Streamlit secrets or environment."
                )
                st.stop()

            # Set envs used by some SDKs
            os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
            if AZURE_OPENAI_ENDPOINT:
                os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

            # Construct AzureChatOpenAI — arguments vary by package version; this is a best-effort config.
            try:
                llm = AzureChatOpenAI(
                    azure_deployment=AZURE_DEPLOYMENT_NAME,
                    api_version=AZURE_OPENAI_VERSION or "2024-07-01-preview",
                    temperature=0.2,
                    max_tokens=600,
                )
                return llm
            except Exception as e:
                st.warning(f"Could not instantiate AzureChatOpenAI: {e}. Trying generic ChatOpenAI fallback.")

        # Fallback to ChatOpenAI
        from langchain.chat_models import ChatOpenAI

        # Use OPENAI_API_KEY if present, otherwise try Azure key for compatibility
        openai_key = os.getenv("OPENAI_API_KEY") or AZURE_OPENAI_API_KEY
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        # model_name may vary by your OpenAI access; change as needed
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini", max_tokens=600)
        return llm
    except Exception as e:
        st.error(f"Failed to create LLM instance: {e}\n{traceback.format_exc()}")
        st.stop()


class HealthcareAssistant:
    """Healthcare Assistant with RAG capabilities"""

    def __init__(self):
        """Initialize the Healthcare Assistant"""
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.initialize_components()

    def initialize_components(self):
        """Initialize all components"""
        try:
            # Initialize embeddings (CPU-friendly)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

            # Create documents
            documents = self.create_documents()

            # Initialize FAISS vector store (in-memory)
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Build LLM
            self.llm = build_llm()

            # Create prompt template
            prompt_template = (
                "You are a professional healthcare AI assistant. Use the following context to answer the question.\n"
                "Always provide disclaimers about seeking professional medical advice.\n\n"
                "Context: {context}\n\n"
                "Chat History: {chat_history}\n\n"
                "Question: {question}\n\n"
                "Helpful Answer:"
            )

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )

            # Create ConversationalRetrievalChain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

        except Exception as e:
            st.error(f"Initialization error: {e}\n{traceback.format_exc()}")
            st.stop()

    def create_documents(self) -> List[Document]:
        """Create documents from knowledge base"""
        documents: List[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        for item in MEDICAL_KNOWLEDGE_BASE:
            chunks = text_splitter.split_text(item["content"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"title": item["title"], "source": "Medical KB"}
                )
                documents.append(doc)

        return documents

    def analyze_medicine_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze medicine image (placeholder).

        This function currently returns a safe placeholder message. For production, integrate a vision
        model (Azure Vision, Google Vision, or a hosted image-classification model) and perform
        OCR / visual matching against a medicine database.
        """
        try:
            # Quick attempt: load image and gather metadata (size, mode)
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            mode = img.mode

            analysis = (
                "This is a placeholder image analysis. For reliable medicine identification, "
                "please integrate a vision API.\n"
                f"Image size: {width}x{height}, mode: {mode}.\n"
                "If you can, describe any imprint/marking/text on the pill, and I can try to help using the knowledge base."
            )

            return {
                "image_analysis": analysis,
                "detailed_info": None,
                "sources": []
            }
        except Exception as e:
            return {
                "image_analysis": f"Error analyzing image: {e}",
                "detailed_info": None,
                "sources": []
            }

    def get_response(self, query: str) -> Dict[str, Any]:
        """Get response from RAG system"""
        try:
            # ConversationalRetrievalChain expects a dict with "question" and optionally "chat_history"
            result = self.qa_chain({"question": query})
            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", [])
            }
        except Exception as e:
            # Fallback to direct LLM completion if chain fails
            try:
                # Many LLM wrappers implement generate or __call__ differently; attempt safe fallbacks
                if hasattr(self.llm, "generate"):
                    resp = self.llm.generate([{"role": "user", "content": query}])
                    text = ""
                    if hasattr(resp, "generations"):
                        gens = resp.generations
                        if gens and len(gens) > 0 and len(gens[0]) > 0:
                            # Some langchain wrappers use .text
                            text = getattr(gens[0][0], "text", str(gens[0][0]))
                    if not text:
                        text = str(resp)
                    return {"answer": text, "source_documents": []}

                # Try calling llm directly if callable
                if callable(self.llm):
                    out = self.llm(query)
                    return {"answer": str(out), "source_documents": []}

                return {"answer": f"(LLM fallback) I couldn't run the RAG chain. Your question was: {query}", "source_documents": []}
            except Exception as e2:
                return {"answer": f"Error: {e} | Fallback error: {e2}", "source_documents": []}

    def add_medical_disclaimer(self, response: str) -> str:
        """Add medical disclaimer"""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only. Always consult healthcare professionals for medical advice."
        return response + disclaimer


# === Streamlit UI helpers ===

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "assistant" not in st.session_state:
        with st.spinner("🔄 Initializing Healthcare Assistant..."):
            st.session_state.assistant = HealthcareAssistant()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Healthcare AI Assistant",
        page_icon="🏥",
        layout="wide"
    )

    # Initialize
    initialize_session_state()

    # Header
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("Medical guidance powered by RAG technology")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This AI assistant provides medical information, symptom checking, "
            "and medication guidance using RAG technology."
        )

        st.header("⚠️ Important")
        st.warning(
            "This is for educational purposes only. Always consult healthcare professionals."
        )

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            # Clear memory if possible
            try:
                st.session_state.assistant.memory.clear()
            except Exception:
                pass
            # use new API
            st.rerun()

        st.header("📋 Quick Questions")
        quick_questions = [
            "What are flu symptoms?",
            "Tell me about aspirin",
            "How to manage diabetes?",
            "COVID-19 prevention"
        ]

        for q in quick_questions:
            if st.button(q, key=f"quick_{q}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    response = st.session_state.assistant.get_response(q)
                    answer = st.session_state.assistant.add_medical_disclaimer(response["answer"])
                    sources = [doc.metadata.get("title", "Unknown") for doc in response.get("source_documents", [])]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": list(set(sources)) if sources else []
                    })
                # use new API
                st.rerun()

    # Main chat area
    st.header("💬 Chat with Healthcare Assistant")

    # Display chat history
    display_chat_history()

    # Chat input
    user_query = st.chat_input("Ask about symptoms, medications, or health conditions...")

    if user_query:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        # Get response
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.assistant.get_response(user_query)
            answer_with_disclaimer = st.session_state.assistant.add_medical_disclaimer(response["answer"])

            # Extract sources
            sources = []
            for doc in response.get("source_documents", []):
                title = doc.metadata.get("title", "Unknown")
                if title not in sources:
                    sources.append(title)

            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_with_disclaimer,
                "sources": sources
            })

        # use new API
        st.rerun()


if __name__ == "__main__":
    main()