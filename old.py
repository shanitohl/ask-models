from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Importación corregida
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuración ---
DOCS_DIR = "docs"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- 1. Cargar y Procesar Documentos ---
def load_documents():
    loaders = [
        PyPDFLoader(f"{DOCS_DIR}/Anexos F.pdf"),
        PyPDFLoader(f"{DOCS_DIR}/ANEXOS CHANO HUAMAN 06-03.pdf")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

# --- 2. Crear Base de Datos Vectorial ---
def create_vector_db(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)  # ✅ Nueva versión
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vector_db")
    return db

# --- 3. Cargar Modelo Local ---
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"  # ✅ Adaptable a GPU/CPU
    )
    return tokenizer, model

# --- 4. Generar Respuesta con RAG ---
def rag_answer(question, db, tokenizer, model):
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    [INST] Responde basándote SOLO en este contexto:
    {context}
    
    Pregunta: {question} 
    [/INST]
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # ✅ Usa el dispositivo correcto
    outputs = model.generate(**inputs, max_length=1000)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer.split("[/INST]")[-1].strip()

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("🔵 Cargando documentos...")
    documents = load_documents()
    
    print("🔵 Creando base de datos vectorial...")
    db = create_vector_db(documents)
    
    print("🔵 Cargando modelo local...")
    tokenizer, model = load_local_model()
    
    question = "¿Como puedes ayudarme?"
    print(f"\n🔵 Pregunta: {question}")
    answer = rag_answer(question, db, tokenizer, model)
    print(f"\n🟢 Respuesta: {answer}")