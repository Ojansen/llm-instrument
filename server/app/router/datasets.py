import logfire
from llama_index.core import Document

from app.types import AgentInterface
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from app.utils.vector_store import VectorStore


class Datasets:
    def __init__(self, llm: AgentInterface):
        self._llm = llm
        self._vector_store = VectorStore(llm=self._llm)

    def generator(self, limit=1, num_questions_per_chunk=1, return_type="dataframe"):
        all_docs = []
        offset = None
        while True:
            response = self._vector_store.client.scroll(
                collection_name=self._vector_store.collection_name,
                offset=offset,
                limit=limit,
                with_payload=True,
            )
            for point in response[0]:
                text = ""
                doc_id = ""
                metadata = {}

                # Ensure payload is present and is a dict
                if point.payload and isinstance(point.payload, dict):
                    text = point.payload.get("text", "")
                    # Remove "text" from metadata
                    metadata = {k: v for k, v in point.payload.items() if k != "text"}
                if point.id is not None:
                    doc_id = str(point.id)
                else:
                    continue  # skip if id is None

                all_docs.append(Document(text=text, doc_id=doc_id, metadata=metadata))
            if response[1] is None:
                break
            offset = response[1]

        # return f"Loaded {len(all_docs)} documents from Qdrant."
        generator = RagDatasetGenerator(
            all_docs,
            llm=self._llm.agent,
            num_questions_per_chunk=num_questions_per_chunk,
            show_progress=True,
        )

        with logfire.span("Dataset generated"):
            logfire.info(
                "details",
                nodes=len(generator.nodes),
                questions_per_chunk=num_questions_per_chunk,
                limit=limit,
            )

        dataset = generator.generate_dataset_from_nodes()

        if return_type == "dataframe":
            return dataset.to_pandas()

        if return_type == "raw":
            return dataset
        return dataset
