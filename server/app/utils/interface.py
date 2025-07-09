import gradio as gr

from app.llm.lmstudio import LmStudio
from app.router.metrics import Metrics
from app.types import AgentInterface
from app.utils.vector_store import VectorStore


class Interface:
    def __init__(self, llm: AgentInterface):
        self._llm = llm
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

                gr.Interface(
                    fn=self._metrics.faithfulness,
                    inputs=["text"],
                    outputs=["text"],
                    title="Faithfulness",
                )
        return block

    def _vector_store_interface(self):
        vs = VectorStore(llm=self._llm, collection_name="test_vectors")
        with gr.Blocks() as block:
            gr.Markdown("# Vector store settings")
            with gr.Row():
                gr.Textbox("test_vectors", label="Collection name", interactive=False)
                gr.Button("Clear Vector Store").click(fn=vs.index().clear)
            with gr.Column():
                gr.Interface(
                    fn=vs.upload_and_index,
                    inputs=gr.Files(label="Upload one or more text files"),
                    outputs="text",
                    title="Upload & Index File to Qdrant",
                )
                gr.Interface(
                    fn=vs.llm_query,
                    inputs=["text"],
                    outputs=["text"],
                    title="Query vector store with LLM response",
                )

        return block
