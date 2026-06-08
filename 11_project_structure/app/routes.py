from fastapi import APIRouter, HTTPException
from models import QuestionRequest
from rag import rag_pipeline

router = APIRouter()

@router.post("/ask")
def ask(data: QuestionRequest):
    if data.question == "":
        raise HTTPException(
            status_code= 400,
            detail= "Question cannot be empty!"
        )
    return rag_pipeline(data.question)

@router.get('/health')
def health():
    return{
        "status": "healthy"
    }

