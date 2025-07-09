import uuid
import os

import logfire
import qdrant_client
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.qdrant import QdrantReader
from qdrant_client.models import PointStruct, VectorParams

from app.types import AgentInterface


class VectorStore:
    def __init__(self, llm: AgentInterface, collection_name="test_vectors"):
        self._embedding_size = 1024
        self._chunk_size = 400
        self.collection_name = collection_name
        self._llm = llm
        self._client = qdrant_client.QdrantClient(
            # url="http://<host>:<port>"
            host="qdrant",
            port=6333,
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
        )
        self.reader = QdrantReader(host="qdrant", port=6333)

    def get_documents(self, text: str, limit=5):
        embedding = self._llm.embed(text)
        documents = self.reader.load_data(
            self.collection_name, query_vector=embedding, limit=limit
        )
        return documents

    def index(self):
        return QdrantVectorStore(
            client=self._client, collection_name=self.collection_name
        )

    def query_engine(self):
        return self.index().as_query_engine(use_async=True)

    def find_or_create_collection(self):
        try:
            self._client.get_collection(self.collection_name)
        except:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_size, distance="Cosine"
                ),
            )

    def upload_and_index(self, files):
        splitter = SentenceSplitter(chunk_size=self._chunk_size)  # Or higher, as needed

        self.find_or_create_collection()
        # file.name is the path to the uploaded file
        try:
            for file in files:
                with open(file.name, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = splitter.split_text(text)
                print(chunks)
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    embedding = self._llm.embed(chunk)

                    # Store in Qdrant (same as before)
                    self._client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "filename": os.path.basename(file.name),
                                    "chunk_index": i,
                                    "text": chunk,
                                },
                            )
                        ],
                    )
            return f"File '{os.path.basename(file.name)}' indexed in Qdrant."
        except Exception as e:
            return f"Error: {e}"
        finally:
            # Clean up temp file Gradio creates
            try:
                os.remove(files)
            except Exception:
                pass

    def query(self, query):
        return self.index().query(
            VectorStoreQuery(
                similarity_top_k=5,
                query_embedding=self._llm.embed(query),
            )
        )

    def llm_query(self, query):
        nodes = self.query(query)

        with logfire.span("Retrieved nodes"):
            logfire.info(
                "documents",
                similarities=nodes.similarities,
                ids=nodes.ids,
                filenames=[node.metadata for node in nodes.nodes],
            )

        context = "\n\n".join([node.text for node in nodes.nodes])
        # Step 3: Create prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        # Step 4: Query the LLM
        response = self._llm.inference(prompt)

        return response
