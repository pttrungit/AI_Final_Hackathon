"""
Healthcare AI Assistant with RAG and Medicine Image Recognition
A domain-specific intelligent assistant for medical guidance, symptom checking, medication information, and medicine image analysis
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import base64
from PIL import Image
import io

# Vector Database and RAG imports - FIXED FOR STREAMLIT CLOUD
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import HumanMessage, SystemMessage

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

# Medical Knowledge Base (Sample data - in production, this would be from medical databases)
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
        Appearance: Usually white round tablets, may have "ASPIRIN" or dosage imprinted."""
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
        Common brands: Tylenol (red and white capsules or white tablets), Panadol (white tablets).
        Appearance: Often white round or oblong tablets, capsules may be red/white or blue/white.
        Vietnamese brands: Panadol Extra (red-white tablets), Hapacol (blue-white capsules), Efferalgan (white effervescent tablets).
        Forms available: Regular tablets (500mg, 650mg), Extra strength (1000mg), Effervescent tablets, Syrup for children.
        Combinations: Often combined with caffeine (Panadol Extra), codeine (for stronger pain relief).
        Storage: Store at room temperature, away from moisture and heat.
        Warning: Avoid alcohol when taking paracetamol. Check other medications as many contain paracetamol.
        Safe for: Pregnant women (consult doctor), children (with appropriate dosing), elderly.
        Onset of action: 30-60 minutes. Duration: 4-6 hours.
        Do not exceed recommended dose - overdose can cause severe liver damage even without immediate symptoms."""
    },
    {
        "title": "Amoxicillin",
        "content": """Amoxicillin is a penicillin antibiotic for bacterial infections.
        Uses: ear infections, strep throat, pneumonia, skin infections, UTIs.
        Dosage: typically 250-500mg every 8 hours or 500-875mg every 12 hours.
        Side effects: nausea, diarrhea, rash. Allergic reactions possible.
        Appearance: Usually pink and white or red and yellow capsules. Tablets may be white or pink."""
    },
    {
        "title": "Omeprazole",
        "content": """Omeprazole is a proton pump inhibitor for acid reflux.
        Uses: GERD, stomach ulcers, heartburn, H. pylori treatment.
        Dosage: 20-40mg once daily before meals.
        Side effects: headache, nausea, diarrhea, stomach pain.
        Common brands: Prilosec (purple and white capsules), generic often beige or pink capsules.
        Take 30-60 minutes before first meal of the day."""
    },
    {
        "title": "Allergies",
        "content": """Allergies occur when immune system reacts to foreign substances. 
        Common allergens: pollen, dust mites, pet dander, foods, insect stings. 
        Symptoms: sneezing, itching, runny nose, watery eyes, rashes, swelling. 
        Treatment: antihistamines (loratadine, cetirizine), nasal steroids, avoid triggers. 
        Severe reactions (anaphylaxis) require immediate epinephrine and emergency care."""
    },
    {
        "title": "Headache and Migraine",
        "content": """Headaches have various causes: tension, cluster, migraine. 
        Migraine symptoms: throbbing pain, nausea, sensitivity to light/sound, aura. 
        Triggers: stress, foods, hormonal changes, sleep changes. 
        Treatment: OTC pain relievers for mild headaches, triptans for migraines, preventive medications. 
        Red flags requiring immediate care: sudden severe headache, fever, neurological symptoms."""
    },
    {
        "title": "COVID-19",
        "content": """COVID-19 is a respiratory illness caused by SARS-CoV-2 virus. 
        Symptoms: fever, cough, shortness of breath, fatigue, body aches, loss of taste/smell. 
        Prevention: vaccination, masks in crowded spaces, hand hygiene, social distancing when needed. 
        Treatment: supportive care for mild cases, antivirals for high-risk patients, hospitalization for severe cases. 
        Testing: PCR or rapid antigen tests available."""
    }
]

class HealthcareAssistant:
    """Main Healthcare Assistant class with RAG capabilities and image recognition"""
    
    def __init__(self):
        """Initialize the Healthcare Assistant with embeddings and vector store"""
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.vision_llm = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all components of the RAG system"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create documents from knowledge base
            documents = self.create_documents()
            
            # Use temporary directory for ChromaDB in Streamlit Cloud
            import tempfile
            temp_dir = tempfile.mkdtemp()
            
            # Initialize vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=temp_dir
            )
            
            # Initialize Azure OpenAI with correct configuration
            os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
            os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
            
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_VERSION,
                temperature=0.3,
                max_tokens=500,
                model="GPT-4o-mini"
            )
            
            # Initialize vision model for image analysis
            self.vision_llm = AzureChatOpenAI(
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_VERSION,
                temperature=0.3,
                max_tokens=300,
                model="GPT-4o-mini"
            )
            
            # Create custom prompt
            prompt_template = """You are a professional healthcare AI assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say so. Always provide disclaimers about seeking professional medical advice.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Helpful Answer (include relevant medical information and always remind users to consult healthcare professionals for serious concerns):"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Create conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
    
    def create_documents(self) -> List[Document]:
        """Create documents from the medical knowledge base"""
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        for item in MEDICAL_KNOWLEDGE_BASE:
            chunks = text_splitter.split_text(item["content"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"title": item["title"], "source": "Medical Knowledge Base"}
                )
                documents.append(doc)
        
        return documents
    
    def analyze_medicine_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze medicine image using vision capabilities"""
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create message with image
            messages = [
                SystemMessage(content="""You are a medication identification expert. Analyze the medicine image and identify:
                1. The likely medication name (generic and brand if visible)
                2. Dosage if visible
                3. Form (tablet, capsule, etc.)
                4. Color and shape
                5. Any visible markings or imprints
                
                Be careful and mention if you're not certain. Always advise verification with a pharmacist."""),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Please identify this medication from the image. Describe what you see and provide possible identification."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                )
            ]
            
            # Get response from vision model
            response = self.vision_llm.invoke(messages)
            
            # Extract medication name for further search
            medicine_info = response.content
            
            # Search knowledge base for more details
            if medicine_info:
                # Extract potential medicine name (simplified extraction)
                search_query = f"Tell me about the medication identified as: {medicine_info[:100]}"
                detailed_info = self.get_response(search_query)
                
                return {
                    "image_analysis": medicine_info,
                    "detailed_info": detailed_info["answer"],
                    "sources": detailed_info.get("source_documents", [])
                }
            
            return {
                "image_analysis": medicine_info,
                "detailed_info": None,
                "sources": []
            }
            
        except Exception as e:
            return {
                "image_analysis": f"Error analyzing image: {str(e)}",
                "detailed_info": None,
                "sources": []
            }
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Get response from the RAG system with fallback to GPT"""
        try:
            # First try to get response from knowledge base
            result = self.qa_chain({"question": query})
            
            # Check if the response indicates no information found
            answer_lower = result["answer"].lower()
            no_info_indicators = [
                "i don't know",
                "i don't have information",
                "not in my knowledge",
                "cannot find",
                "no information available",
                "beyond my knowledge",
                "don't have specific information",
                "not found in",
                "i cannot provide"
            ]
            
            # If no good information in knowledge base, query GPT directly
            if any(indicator in answer_lower for indicator in no_info_indicators):
                # Use GPT to get information
                gpt_response = self.get_gpt_medical_info(query)
                
                # Combine response with source indication
                combined_answer = f"**Information from medical database:**\n\n{gpt_response}\n\n" \
                                f"*Note: This information was retrieved from general medical knowledge " \
                                f"as it was not found in the local knowledge base.*"
                
                return {
                    "answer": combined_answer,
                    "source_documents": [],
                    "source_type": "GPT-4"
                }
            
            # If found in knowledge base, return as normal
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "source_type": "Knowledge Base"
            }
            
        except Exception as e:
            # If error, try GPT as fallback
            try:
                gpt_response = self.get_gpt_medical_info(query)
                return {
                    "answer": f"**Information from medical database:**\n\n{gpt_response}",
                    "source_documents": [],
                    "source_type": "GPT-4 (Fallback)"
                }
            except:
                return {
                    "answer": f"Error processing query: {str(e)}",
                    "source_documents": [],
                    "source_type": "Error"
                }
    
    def get_gpt_medical_info(self, query: str) -> str:
        """Get medical information directly from GPT when not in knowledge base"""
        try:
            prompt = f"""You are a medical information expert. Please provide accurate, detailed medical information about the following query:
            
            Query: {query}
            
            Please include:
            1. Definition/Overview
            2. Symptoms (if applicable)
            3. Causes (if applicable)
            4. Treatment options
            5. Dosage information (if medication)
            6. Side effects (if medication)
            7. Precautions and warnings
            8. When to seek medical help
            
            Provide comprehensive, medically accurate information."""
            
            messages = [
                {"role": "system", "content": "You are a knowledgeable medical information assistant. Provide accurate, detailed medical information."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Unable to retrieve information from medical database: {str(e)}"
    
    def add_medical_disclaimer(self, response: str) -> str:
        """Add medical disclaimer to responses"""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for medical concerns. For medication identification, always verify with a licensed pharmacist."
        return response + disclaimer

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "assistant" not in st.session_state:
        st.session_state.assistant = HealthcareAssistant()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def display_chat_history():
    """Display chat history in the UI"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                if message.get("image"):
                    st.image(message["image"], caption="Uploaded medicine image", width=200)
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Healthcare AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🏥 Healthcare AI Assistant with Medicine Recognition")
    st.markdown("Your intelligent medical guidance companion powered by RAG technology and image analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This AI assistant provides medical information, symptom checking, "
            "medication guidance, and can identify medicines from images using RAG and vision AI."
        )
        
        st.header("🔧 Features")
        st.markdown(
            """
            - **📷 Medicine Image Analysis**: Upload medicine photos for identification
            - **Symptom Analysis**: Describe symptoms for initial guidance
            - **Medication Info**: Learn about common medications
            - **Disease Information**: Get details about various conditions
            - **Health Tips**: General wellness recommendations
            """
        )
        
        st.header("⚠️ Important")
        st.warning(
            "This assistant provides general information only. "
            "Always verify medication identification with a pharmacist. "
            "Consult healthcare professionals for medical decisions."
        )
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.assistant.memory.clear()
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat with Healthcare Assistant")
        
        # Medicine Image Upload Section
        with st.expander("📷 Upload Medicine Image for Identification", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload a clear image of the medicine (pill, tablet, capsule)",
                type=['png', 'jpg', 'jpeg'],
                help="Take a clear photo showing any markings, imprints, or text on the medicine"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Medicine", width=300)
                
                if st.button("🔍 Analyze Medicine"):
                    with st.spinner("🤖 Analyzing medicine image..."):
                        # Convert image to bytes - handle RGBA images
                        img_byte_arr = io.BytesIO()
                        
                        # Convert RGBA to RGB if necessary
                        if image.mode == 'RGBA':
                            # Create a white background
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                            image = background
                        elif image.mode not in ['RGB', 'L']:
                            # Convert other modes to RGB
                            image = image.convert('RGB')
                        
                        image.save(img_byte_arr, format='JPEG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Analyze image
                        result = st.session_state.assistant.analyze_medicine_image(img_bytes)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": "Please identify this medicine from the image.",
                            "image": image
                        })
                        
                        # Prepare response
                        response_text = f"**Image Analysis Results:**\n\n{result['image_analysis']}"
                        
                        if result['detailed_info']:
                            response_text += f"\n\n**Additional Information from Knowledge Base:**\n{result['detailed_info']}"
                        
                        response_with_disclaimer = st.session_state.assistant.add_medical_disclaimer(response_text)
                        
                        # Extract sources
                        sources = []
                        for doc in result.get('sources', []):
                            title = doc.metadata.get("title", "Unknown")
                            if title not in sources:
                                sources.append(title)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_with_disclaimer,
                            "sources": sources
                        })
                    
                    st.rerun()
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # User input for text queries
        user_query = st.chat_input("Ask about symptoms, medications, or health conditions...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            # Get assistant response
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.assistant.get_response(user_query)
                
                # Add disclaimer to medical responses
                answer_with_disclaimer = st.session_state.assistant.add_medical_disclaimer(
                    response["answer"]
                )
                
                # Extract source titles
                sources = []
                source_type = response.get("source_type", "Unknown")
                
                if source_type == "Knowledge Base":
                    for doc in response["source_documents"]:
                        title = doc.metadata.get("title", "Unknown")
                        if title not in sources:
                            sources.append(title)
                elif source_type in ["GPT-4", "GPT-4 (Fallback)"]:
                    sources = ["GPT-4 Medical Database"]
                
                # Add source type indicator to response
                if source_type != "Knowledge Base":
                    answer_with_disclaimer = answer_with_disclaimer + f"\n\n📊 **Source**: {source_type}"
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_with_disclaimer,
                    "sources": sources
                })
            
            st.rerun()
    
    with col2:
        st.header("📋 Quick Questions")
        
        quick_questions = [
            "What are the symptoms of flu?",
            "Tell me about aspirin dosage",
            "How to manage diabetes?",
            "What causes headaches?",
            "COVID-19 prevention tips",
            "Allergy treatment options",
            "High blood pressure info",
            "Paracetamol vs Ibuprofen?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.user_input = question
                # Trigger the same flow as manual input
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                response = st.session_state.assistant.get_response(question)
                answer_with_disclaimer = st.session_state.assistant.add_medical_disclaimer(
                    response["answer"]
                )
                
                # Handle sources based on source type
                source_type = response.get("source_type", "Unknown")
                if source_type == "Knowledge Base":
                    sources = [doc.metadata.get("title", "Unknown") 
                              for doc in response["source_documents"]]
                else:
                    sources = ["GPT-4 Medical Database"]
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_with_disclaimer,
                    "sources": list(set(sources))
                })
                st.rerun()
        
        st.header("💊 Common Medicines")
        st.info(
            """
            **Can Identify:**
            - Aspirin
            - Ibuprofen (Advil/Motrin)
            - Paracetamol/Acetaminophen
            - Metformin
            - Amoxicillin
            - Omeprazole
            - And more...
            
            Upload clear images showing pill markings!
            """
        )

if __name__ == "__main__":
    main()
