"""
Healthcare AI Assistant with RAG and Medicine Image Recognition
Streamlit Cloud Compatible Version
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import base64
from PIL import Image
import io
import tempfile

# Fixed imports for Streamlit Cloud
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Azure OpenAI Configuration with Streamlit Secrets
try:
    # For Streamlit Cloud deployment
    AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
    AZURE_OPENAI_VERSION = st.secrets["AZURE_OPENAI_VERSION"]
except:
    # For local development
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "sk-GQ1M8LeXBpca_FNVSNDByA")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "GPT-4o-mini")
    AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-07-01-preview")

# Medical Knowledge Base
MEDICAL_KNOWLEDGE_BASE = [
    {
        "title": "Common Cold",
        "content": """The common cold is a viral infection of the upper respiratory tract. 
        Symptoms include: runny nose, sneezing, sore throat, cough, congestion, slight body aches, 
        mild headache, low-grade fever. Treatment: rest, fluids, over-the-counter pain relievers. 
        Prevention: hand washing, avoid touching face, avoid close contact with sick people."""
    },
    {
        "title": "Influenza (Flu)",
        "content": """Influenza is a respiratory illness caused by flu viruses. 
        Symptoms: fever, chills, muscle aches, cough, congestion, runny nose, headaches, fatigue. 
        Treatment: antiviral drugs if started early, rest, fluids. Prevention: annual flu vaccine, 
        hand hygiene, avoid close contact with sick individuals."""
    },
    {
        "title": "Hypertension",
        "content": """High blood pressure (hypertension) is when blood pressure is consistently too high. 
        Normal: Less than 120/80 mmHg. Elevated: 120-129/<80. Stage 1: 130-139/80-89. Stage 2: 140/90 or higher. 
        Treatment: lifestyle changes (diet, exercise, weight loss), medications (ACE inhibitors, beta-blockers, 
        diuretics). Complications if untreated: heart disease, stroke, kidney problems."""
    },
    {
        "title": "Diabetes Type 2",
        "content": """Type 2 diabetes is a chronic condition affecting blood sugar regulation. 
        Symptoms: increased thirst, frequent urination, hunger, fatigue, blurred vision, slow-healing sores. 
        Management: blood sugar monitoring, healthy diet, regular exercise, medications (metformin, insulin). 
        Complications: heart disease, nerve damage, kidney damage, eye problems."""
    },
    {
        "title": "Aspirin",
        "content": """Aspirin (acetylsalicylic acid) is a common pain reliever and anti-inflammatory. 
        Uses: pain relief, fever reduction, anti-inflammatory, blood clot prevention. 
        Dosage: 325-650mg every 4 hours for pain. Low dose (81mg) daily for heart disease prevention. 
        Side effects: stomach upset, bleeding risk. Contraindications: bleeding disorders, children with fever.
        Appearance: Usually white round tablets, may have ASPIRIN or dosage imprinted."""
    },
    {
        "title": "Ibuprofen",
        "content": """Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). 
        Uses: pain relief, fever reduction, inflammation reduction. 
        Dosage: 200-400mg every 4-6 hours, maximum 1200mg/day without prescription. 
        Side effects: stomach upset, dizziness, rash. Warnings: may increase heart attack/stroke risk with long-term use.
        Common brands: Advil (red/brown coated tablets), Motrin (orange tablets). Usually oval or round shaped."""
    },
    {
        "title": "Metformin",
        "content": """Metformin is a first-line medication for type 2 diabetes. 
        Action: decreases glucose production in liver, improves insulin sensitivity. 
        Dosage: typically start 500mg once/twice daily, can increase to 2000mg/day. 
        Side effects: nausea, diarrhea, stomach upset. Rare but serious: lactic acidosis. 
        Take with meals to reduce GI side effects. Usually white oval or round tablets with dosage imprinted."""
    },
    {
        "title": "Paracetamol (Acetaminophen)",
        "content": """Paracetamol/Acetaminophen is a pain reliever and fever reducer.
        Uses: mild to moderate pain, fever reduction. Does not reduce inflammation.
        Dosage: 500-1000mg every 4-6 hours, maximum 4000mg/day for adults.
        Side effects: rare when used correctly. Overdose can cause liver damage.
        Common brands: Tylenol, Panadol. Appearance: Often white round or oblong tablets.
        Warning: Avoid alcohol when taking paracetamol. Check other medications as many contain paracetamol."""
    },
    {
        "title": "COVID-19",
        "content": """COVID-19 is a respiratory illness caused by SARS-CoV-2 virus. 
        Symptoms: fever, cough, shortness of breath, fatigue, body aches, loss of taste/smell. 
        Prevention: vaccination, masks in crowded spaces, hand hygiene, social distancing when needed. 
        Treatment: supportive care for mild cases, antivirals for high-risk patients, hospitalization for severe cases."""
    }
]

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
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create documents
            documents = self.create_documents()
            
            # Initialize FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Set environment variables
            os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
            os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
            
            # Initialize Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_VERSION,
                temperature=0.3,
                max_tokens=500,
                model="GPT-4o-mini"
            )
            
            # Create prompt template
            prompt_template = """You are a professional healthcare AI assistant. Use the following context to answer the question.
            Always provide disclaimers about seeking professional medical advice.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Helpful Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Create QA chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()
    
    def create_documents(self) -> List[Document]:
        """Create documents from knowledge base"""
        documents = []
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
        """Analyze medicine image"""
        try:
            # Encode image
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create prompt for image analysis
            prompt = f"""Analyze this medicine image and identify:
            1. Medication name (generic and brand if visible)
            2. Dosage if visible
            3. Form (tablet, capsule, etc.)
            4. Color and shape
            5. Any visible markings
            
            Be careful and advise verification with a pharmacist."""
            
            # For now, return a simple analysis (Vision API would go here)
            # In production, you would call the vision model here
            analysis = "Image analysis: Medicine identification requires vision API integration. Please describe the medicine for text-based assistance."
            
            return {
                "image_analysis": analysis,
                "detailed_info": None,
                "sources": []
            }
            
        except Exception as e:
            return {
                "image_analysis": f"Error: {str(e)}",
                "detailed_info": None,
                "sources": []
            }
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Get response from RAG system"""
        try:
            result = self.qa_chain({"question": query})
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", [])
            }
        except Exception as e:
            # Fallback to direct LLM
            try:
                response = self.llm.invoke([{"role": "user", "content": query}])
                return {
                    "answer": response.content,
                    "source_documents": []
                }
            except:
                return {
                    "answer": f"Error: {str(e)}",
                    "source_documents": []
                }
    
    def add_medical_disclaimer(self, response: str) -> str:
        """Add medical disclaimer"""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only. Always consult healthcare professionals for medical advice."
        return response + disclaimer

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
            "This is for educational purposes only. "
            "Always consult healthcare professionals."
        )
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.assistant.memory.clear()
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
                # Process quick question
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    response = st.session_state.assistant.get_response(q)
                    answer = st.session_state.assistant.add_medical_disclaimer(response["answer"])
                    sources = [doc.metadata.get("title", "Unknown") for doc in response["source_documents"]]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": list(set(sources)) if sources else []
                    })
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
        
        st.rerun()

if __name__ == "__main__":
    main()
