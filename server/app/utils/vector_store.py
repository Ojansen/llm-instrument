import uuid

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.qdrant import QdrantReader
import qdrant_client
import os
from qdrant_client.models import PointStruct, VectorParams

from app.router.metrics import AgentInterface


class VectorStore:
    def __init__(self, llm: AgentInterface, collection_name="test_vectors"):
        self.embedding_size = 1024
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
                    size=self.embedding_size, distance="Cosine"
                ),
            )

    def upload_and_index(self, file):
        self.find_or_create_collection()
        # file.name is the path to the uploaded file
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
            embedding = self._llm.embed(text)
            # Store in Qdrant (same as before)
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={"filename": os.path.basename(file.name), "text": text},
                    )
                ],
            )
            return f"File '{os.path.basename(file.name)}' indexed in Qdrant."
        except Exception as e:
            return f"Error: {e}"
        finally:
            # Clean up temp file Gradio creates
            try:
                os.remove(file.name)
            except Exception:
                pass

    def query(self, query):
        return str(
            self.index().query(
                VectorStoreQuery(
                    query_embedding=self._llm.embed(query),
                )
            )
        )
