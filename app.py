from fastapi import FastAPI
from pydantic import BaseModel
import faq as faq
import uvicorn
import gradio as gr

app = FastAPI()


class Request(BaseModel):
    question: str
    sheet_url: str
    page_content_column: str
    k: int


@app.post("/api/v1/ask")
async def ask_api(request: Request):
    return ask(
        request.sheet_url, request.page_content_column, request.k, request.question
    )


def ask(sheet_url: str, page_content_column: str, k: int, question: str):
    vectordb = faq.load_vectordb(sheet_url, page_content_column)
    result = faq.similarity_search(vectordb, question, k=k)
    return result


iface = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(label="Google Sheet URL"),
        gr.Textbox(label="Question Column"),
        gr.Slider(1, 5, step=1, label="K"),
        gr.Textbox(label="Question"),
    ],
    outputs=[gr.JSON(label="Answer")],
    allow_flagging="never",
)

app = gr.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
