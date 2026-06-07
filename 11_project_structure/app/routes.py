from fastapi import APIRouter
from models import QuestionRequest
from rag import rag_pipeline

router = APIRouter()

@router.post("/ask")
def ask(data: QuestionRequest):
    return rag_pipeline(data.question)