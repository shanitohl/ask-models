from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import List

class RAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.vector_db = None
        self.embeddings_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def load_documents(self, docs_dir: str) -> List:
        """Carga y divide documentos PDF/TXT de una carpeta"""
        loaders = [
            PyPDFLoader(os.path.join(docs_dir, "Anexos F.pdf")),
            PyPDFLoader(os.path.join(docs_dir, "ANEXOS CHANO HUAMAN 06-03.pdf"))
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return text_splitter.split_documents(documents)

    def initialize_models(self, docs_dir: str = "docs"):
        """Inicializa el modelo de lenguaje y la base de datos vectorial"""
        # 1. Cargar modelo LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # 2. Cargar embeddings y base vectorial
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        db_path = "vector_db"
        
        if not os.path.exists(db_path):
            documents = self.load_documents(docs_dir)
            self.vector_db = FAISS.from_documents(documents, embeddings)
            self.vector_db.save_local(db_path)
        else:
            self.vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    def generate_answer(self, question: str, use_rag: bool = True) -> str:
        """Genera respuesta usando RAG"""
        if not self.vector_db or not self.model:
            raise RuntimeError("Modelo no inicializado")
        
        # Modo RAG (con documentos)
        if use_rag:
            print("Loading documents...")
            docs = self.vector_db.similarity_search(question, k=3)
            print(docs)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
            [INST] Responde basándote SOLO en este contexto:
            {context}
            
            Pregunta: {question} 
            [/INST]
            """
        # Modo chat directo
        else:
            print("talk to me")
            prompt = f"[INST] {question} [/INST]"
        
        # Generación de respuesta
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, 
                                        max_length=2048,
                                        temperature=0.7,  # Controla la creatividad
                                        do_sample=True,
                                        top_p=0.9,
                                        num_return_sequences=1)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(full_response)

        # Extraer solo la parte después de [/INST]
        if "[/INST]" in full_response:
            answer = full_response.split("[/INST]")[-1].strip()
        else:
            answer = full_response
        
        # Limpieza adicional
        answer = answer.replace("\n", " ").strip()  # Eliminar saltos de línea múltiples
        answer = ' '.join(answer.split())  # Eliminar espacios duplicados
        
        return answer