from fastapi import FastAPI
from pydantic import BaseModel
import faq as faq
import util as util
import uvicorn
import gradio as gr

app = FastAPI()


class AskRequest(BaseModel):
    question: str
    sheet_url: str
    page_content_column: str
    k: int


@app.post("/api/v1/ask")
async def ask_api(request: AskRequest):
    return ask(
        request.sheet_url, request.page_content_column, request.k, request.question
    )


@app.post("/api/v2/ask")
async def ask_api(request: AskRequest):
    util.SPLIT_PAGE_BREAKS = True
    vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
    documents = faq.similarity_search(vectordb, request.question, k=request.k)
    df_doc = util.transform_documents_to_dataframe(documents)
    df_filter = util.remove_duplicates_by_column(df_doc, "ID")
    return util.dataframe_to_dict(df_filter)


@app.delete("/api/v1/")
async def delete_vectordb_api():
    return delete_vectordb()


def ask(sheet_url: str, page_content_column: str, k: int, question: str):
    util.SPLIT_PAGE_BREAKS = False
    vectordb = faq.load_vectordb(sheet_url, page_content_column)
    result = faq.similarity_search(vectordb, question, k=k)
    return result


def delete_vectordb():
    faq.delete_vectordb()


with gr.Blocks() as block:
    sheet_url = gr.Textbox(label="Google Sheet URL")
    page_content_column = gr.Textbox(label="Question Column")
    k = gr.Slider(2, 5, step=1, label="K")
    question = gr.Textbox(label="Question")
    ask_button = gr.Button("Ask")
    answer_output = gr.JSON(label="Answer")
    delete_button = gr.Button("Delete Vector DB")
    ask_button.click(
        ask,
        inputs=[sheet_url, page_content_column, k, question],
        outputs=answer_output,
    )
    delete_button.click(delete_vectordb)

app = gr.mount_gradio_app(app, block, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
