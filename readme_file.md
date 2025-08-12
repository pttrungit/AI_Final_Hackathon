# Healthcare AI Assistant with RAG

## 📋 Overview
A domain-specific intelligent AI assistant for healthcare guidance, leveraging Retrieval-Augmented Generation (RAG) techniques. The assistant provides medical information, symptom checking, and medication guidance while maintaining context through conversation memory.

## 🚀 Features

### Core Capabilities
- **Symptom Analysis**: Analyze and provide information about various symptoms
- **Medication Information**: Detailed information about common medications, dosages, and side effects
- **Disease Information**: Comprehensive details about various medical conditions
- **Conversational Memory**: Maintains context throughout the conversation
- **Source Attribution**: Shows which medical knowledge sources were used for answers
- **Medical Disclaimers**: Automatic addition of appropriate medical disclaimers

### Technical Features
- **RAG Implementation**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Vector Database**: ChromaDB for efficient similarity search
- **Embeddings**: HuggingFace sentence transformers for text vectorization
- **Azure OpenAI Integration**: GPT-4o-mini for natural language generation
- **Streamlit UI**: User-friendly web interface

## 📦 Requirements

```bash
pip install streamlit
pip install langchain
pip install chromadb
pip install sentence-transformers
pip install openai
pip install tiktoken
```

## 🛠️ Installation

1. **Clone the repository or save the file**
```bash
# Save the main file as healthcare_assistant.py
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run healthcare_assistant.py
```

## 🔧 Configuration

The application uses Azure OpenAI with the following configuration:
- **API Key**: Configured in the application
- **Endpoint**: https://aiportalapi.stu-platform.live/jpe
- **Model**: GPT-4o-mini
- **API Version**: 2024-07-01-preview

## 💻 Usage

### Starting the Application
1. Run `streamlit run healthcare_assistant.py`
2. Open browser at `http://localhost:8501`

### Using the Assistant
1. **Type your question** in the chat input box
2. **Use quick questions** from the sidebar for common queries
3. **View sources** by expanding the sources section under responses
4. **Clear chat history** using the button in the sidebar

### Example Queries
- "What are the symptoms of the common cold?"
- "Tell me about ibuprofen dosage and side effects"
- "How can I manage type 2 diabetes?"
- "What should I know about high blood pressure?"
- "What are the COVID-19 prevention measures?"

## 🏗️ Architecture

### Components

1. **HealthcareAssistant Class**
   - Main orchestrator for the RAG system
   - Manages embeddings, vector store, and LLM
   - Handles conversation memory

2. **Vector Store (ChromaDB)**
   - Stores medical knowledge embeddings
   - Enables semantic search for relevant information

3. **Embeddings (HuggingFace)**
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Converts text to vector representations

4. **Language Model (Azure OpenAI)**
   - GPT-4o-mini for response generation
   - Temperature: 0.3 for balanced creativity/accuracy

5. **Streamlit Interface**
   - Chat interface with history
   - Quick question buttons
   - Source attribution display

### Data Flow
1. User inputs question
2. Question is embedded using HuggingFace
3. Similar documents retrieved from ChromaDB
4. Context + question sent to Azure OpenAI
5. Response generated with medical disclaimer
6. Response displayed with source attribution

## 📊 Medical Knowledge Base

The system includes pre-loaded information about:
- **Common Conditions**: Cold, Flu, Hypertension, Diabetes, Allergies, Headaches, COVID-19
- **Medications**: Aspirin, Ibuprofen, Metformin
- **Symptoms and Treatments**: Comprehensive symptom descriptions and treatment options
- **Prevention Measures**: Preventive care recommendations

## ⚠️ Important Disclaimers

- This assistant provides **general medical information only**
- Not a substitute for professional medical advice
- Always consult qualified healthcare providers
- Not for emergency medical situations
- Information is educational in nature

## 🔒 Privacy & Security

- No personal health information is stored permanently
- Conversation history is session-based only
- Data is processed locally with vector embeddings
- API calls are made securely to Azure OpenAI

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version (3.8+ recommended)

2. **API Connection Issues**
   - Verify Azure OpenAI credentials
   - Check internet connection
   - Ensure API endpoint is accessible

3. **Memory Issues**
   - Clear chat history if responses slow down
   - Restart application if needed

4. **ChromaDB Errors**
   - Delete `./chroma_db` folder and restart
   - Ensure write permissions in directory

## 📈 Performance Optimization

- **Chunk Size**: 500 characters for optimal retrieval
- **Retrieval Count**: Top 3 most relevant documents
- **Temperature**: 0.3 for consistent medical information
- **Max Tokens**: 500 for concise responses

## 🔄 Updates & Maintenance

### Adding New Medical Knowledge
1. Add entries to `MEDICAL_KNOWLEDGE_BASE` list
2. Restart application to rebuild vector store
3. New information will be automatically indexed

### Updating Models
- Change `model_name` in HuggingFaceEmbeddings
- Update Azure OpenAI deployment name if needed

## 📝 License

This project is for educational purposes. Medical information should be verified with professional sources.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example queries
3. Ensure all dependencies are correctly installed
4. Restart the application if experiencing issues

---

**Remember**: This is an AI assistant for educational purposes. Always seek professional medical advice for health concerns.