from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import RAGSystem
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI(title="RAG API")
rag_system = RAGSystem()

# Modelo para las solicitudes
class QuestionRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    message: str

# Evento de inicio
@app.on_event("startup")
async def startup_event():
    rag_system.initialize_models()
    print("✅ Modelo y base de datos vectorial cargados")

# Endpoint principal
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        print("Finding answer...")
        answer = rag_system.generate_answer(request.question, use_rag=True)
        return JSONResponse(content={
            "answer": answer,
            "status": "success",
            "source": "RAG"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat")
async def direct_chat(request: ChatRequest):
    """Endpoint para chat directo con el modelo"""
    try:
        response = rag_system.generate_answer(request.message, use_rag=False)
        return JSONResponse(content={
            "response": response,
            "status": "success",
            "source": "direct-chat"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)