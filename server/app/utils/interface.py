import gradio as gr

from app.database.db import Database
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
        self._database = Database()

    def render(self):
        return gr.TabbedInterface(
            interface_list=[
                self._dashboard(),
                self._metric_interface(),
                self._vector_store_interface(),
                self._dataset_interface(),
                self._database_interface(),
            ],
            tab_names=[
                "Dashboard",
                "Metrics",
                "Vector store",
                "Dataset",
                "Database",
            ],
        )

    @staticmethod
    def _dashboard():
        with gr.Blocks() as block:
            gr.Markdown(
                "# Dashboard \n## Links \n* [otel](http://localhost:16686) \n* [Qdrant](http://localhost:6333/dashboard#/collections)"
            )
        return block

    def _metric_interface(self):
        def similarity_and_correctness(*args, **kwargs):
            sim_results = self._metrics.cosine_similarity(*args, **kwargs)
            result = self._metrics.correctness(*args, **kwargs)
            return f"{sim_results.feedback}  \n\nCorrectness \nScore {result.score} \nDid pass: {result.passing} \nFeedback: {result.feedback}"

        def ragchecker(*args, **kwargs):
            result = self._metrics.ragchecker(*args, **kwargs)
            return result

        def faithfulness(*args, **kwargs):
            result = self._metrics.faithfulness(*args, **kwargs)
            return f"Faithfulness/ Hallucinations \nScore: {result.score} \nDid pass: {result.passing} \nFeedback: {result.feedback}"

        with gr.Blocks() as block:
            with gr.Column():
                gr.Interface(
                    fn=similarity_and_correctness,
                    inputs=[gr.Textbox(label="Prompt"), gr.Textbox(label="Context")],
                    outputs=["text"],
                    title="Similarity",
                )
                gr.Markdown("# Vector store metrics")
                gr.Interface(
                    fn=faithfulness,
                    inputs=["text"],
                    outputs=["text"],
                    title="Faithfulness",
                    description="Evaluates whether a response is faithful to the contexts \n(i.e. whether the response is supported by the contexts or hallucinated.)",
                )
                gr.Interface(
                    fn=ragchecker,
                    inputs=[
                        gr.Textbox(label="Prompt"),
                        gr.Textbox(label="Ground truth"),
                    ],
                    outputs=[gr.JSON()],
                    title="Rag checker",
                    description="![image](https://raw.githubusercontent.com/amazon-science/RAGChecker/refs/heads/main/imgs/ragchecker_metrics.png)",
                    article="[Package](https://github.com/amazon-science/RAGChecker)",
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

    def _database_interface(self):
        with gr.Blocks() as block:

            textbox = gr.Textbox()
            gr.Button("Create Tables").click(
                fn=self._database.create_all, outputs=[textbox]
            )
            gr.Button("Drop all").click(fn=self._database.drop_all, outputs=[textbox])

            gr.Interface(
                fn=self._database.create_project, inputs=["text"], outputs=None
            )

            gr.Interface(
                fn=self._database.create_new_session,
                inputs=[
                    gr.Dropdown(
                        [project.id for project in self._database.all_projects()],
                        label="Project name",
                    )
                ],
                outputs=["text"],
            )

            # gr.Button(value="Update questions").click(fn=)

        return block
