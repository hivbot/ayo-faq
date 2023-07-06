from fastapi import FastAPI
from pydantic import BaseModel
import faq as faq
import util as util
import uvicorn
import gradio as gr
from typing import List, Optional
from fastapi.responses import JSONResponse

app = FastAPI()


class Request(BaseModel):
    question: Optional[str] = "?"
    sheet_url: str
    page_content_column: str
    k: Optional[int] = 20
    reload_collection: Optional[bool] = False
    id_column: Optional[str] = None
    synonyms: Optional[List[List[str]]] = None


@app.post("/api")
async def post_api(request: Request) -> JSONResponse:
    if request.id_column is not None:
        util.SPLIT_PAGE_BREAKS = True
    if request.synonyms is not None:
        util.SYNONYMS = request.synonyms
    vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
    if request.reload_collection:
        faq.delete_vectordb_current_collection(vectordb)
        vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
    documents = faq.similarity_search(vectordb, request.question, k=request.k)
    df_doc = util.transform_documents_to_dataframe(documents)
    if request.id_column is not None:
        df_doc = util.remove_duplicates_by_column(df_doc, request.id_column)
    return JSONResponse(util.dataframe_to_dict(df_doc))


@app.put("/api")
async def put_api(request: Request) -> bool:
    success = False
    if request.id_column is not None:
        util.SPLIT_PAGE_BREAKS = True
    if request.synonyms is not None:
        util.SYNONYMS = request.synonyms
    vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
    if request.reload_collection:
        faq.delete_vectordb_current_collection(vectordb)
        vectordb = faq.load_vectordb(request.sheet_url, request.page_content_column)
        success = True
    return success


@app.delete("/api")
async def delete_vectordb_api() -> None:
    faq.delete_vectordb()


def ask(sheet_url: str, page_content_column: str, k: int, reload_collection: bool, question: str):
    util.SPLIT_PAGE_BREAKS = False
    vectordb = faq.load_vectordb(sheet_url, page_content_column)
    if reload_collection:
        faq.delete_vectordb_current_collection(vectordb)
        vectordb = faq.load_vectordb(sheet_url, page_content_column)
    documents = faq.similarity_search(vectordb, question, k=k)
    df_doc = util.transform_documents_to_dataframe(documents)
    return util.dataframe_to_dict(df_doc), gr.Checkbox.update(False)


with gr.Blocks() as block:
    sheet_url = gr.Textbox(label="Google Sheet URL")
    page_content_column = gr.Textbox(label="Question Column")
    k = gr.Slider(1, 30, step=1, label="K")
    reload_collection = gr.Checkbox(label="Reload Collection?")
    question = gr.Textbox(label="Question")
    ask_button = gr.Button("Ask")
    answer_output = gr.JSON(label="Answer")
    ask_button.click(
        ask,
        inputs=[sheet_url, page_content_column, k, reload_collection, question],
        outputs=[answer_output, reload_collection]
    )

app = gr.mount_gradio_app(app, block, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
