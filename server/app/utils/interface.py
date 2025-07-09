import gradio as gr

from app.llm.lmstudio import LmStudio
from app.router.metrics import Metrics
from app.utils.vector_store import VectorStore


class Interface:
    def __init__(self):
        self._llm = LmStudio(system_prompt="")
        self._metrics = Metrics(llm=self._llm)

    def render(self):
        return gr.TabbedInterface(
            interface_list=[self._metric_interface(), self._vector_store_interface()],
            tab_names=["Metrics", "Vector store"],
        )

    def _metric_interface(self):
        with gr.Blocks() as block:
            with gr.Column():
                gr.Interface(
                    fn=self._metrics.correctness,
                    inputs=["text", "text"],
                    outputs=["text"],
                    title="Correctness",
                )
                gr.Interface(
                    fn=self._metrics.cosine_similarity,
                    inputs=["text", "text"],
                    outputs=["text"],
                    title="Similarity",
                )
        return block

    def _vector_store_interface(self):
        vs = VectorStore(llm=self._llm, collection_name="test_vectors")
        with gr.Blocks() as block:
            with gr.Column():
                gr.Interface(
                    fn=vs.upload_and_index,
                    inputs=gr.File(label="Upload a text file"),
                    outputs="text",
                    title="Upload & Index File to Qdrant",
                )
                gr.Interface(
                    fn=vs.query,
                    inputs=["text"],
                    outputs=["text"],
                    title="Query Vectors",
                )

        return block
