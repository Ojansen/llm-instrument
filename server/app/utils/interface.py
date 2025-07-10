import gradio as gr

from app.router.datasets import Datasets
from app.router.metrics import Metrics
from app.types import AgentInterface
from app.utils.vector_store import VectorStore


class Interface:
    def __init__(self, llm: AgentInterface):
        self._llm = llm
        self._metrics = Metrics(llm=self._llm)
        self._vector_store = VectorStore(llm=self._llm)
        self._datasets = Datasets(llm=self._llm)

    def render(self):
        return gr.TabbedInterface(
            interface_list=[
                self._metric_interface(),
                self._vector_store_interface(),
                self._dataset_interface(),
            ],
            tab_names=["Metrics", "Vector store", "Dataset"],
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
        with gr.Blocks() as block:
            gr.Markdown("# Vector store settings")
            with gr.Row():
                gr.Textbox("test_vectors", label="Collection name", interactive=False)
                gr.Button("Clear Vector Store").click(
                    fn=self._vector_store.index().clear
                )
            with gr.Column():
                gr.Interface(
                    fn=self._vector_store.upload_and_index,
                    inputs=gr.Files(label="Upload one or more text files"),
                    outputs="text",
                    title="Upload & Index File to Qdrant",
                )
                gr.Interface(
                    fn=self._vector_store.llm_query,
                    inputs=["text"],
                    outputs=["text"],
                    title="Query vector store with LLM response",
                )

        return block

    def _dataset_interface(self):
        with gr.Blocks() as block:
            # with gr.Row():
            #     limit = gr.Number(label="Limit", value=1, precision=0)
            #     num_questions = gr.Number(
            #         label="Num Questions per Chunk", value=1, precision=0
            #     )
            gr.Button(value="Create questions").click(
                fn=self._datasets.generator,
                # inputs=[limit, num_questions],
                outputs=gr.Dataframe(label="RAG Dataset"),
            )

        return block
