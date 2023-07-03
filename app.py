from fastapi import FastAPI
from pydantic import BaseModel
import faq as faq
import uvicorn

app = FastAPI()


class Request(BaseModel):
    question: str
    sheet_url: str
    page_content_column: str
    k: int


@app.post("/api/v1/ask")
async def ask(request: Request):
    vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
    result = faq.similarity_search(vectordb, request.question, k=request.k)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
