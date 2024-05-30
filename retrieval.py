from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import QueryBundle
from llama_index.legacy.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from typing import Optional, Any, List

class VectorDBRetriever(BaseRetriever):
    def __init__(
            self,
            vector_store: PGVectorStore,
            embed_model: Any,
            query_mode: str = "default",
            similarity_top_k: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.query_mode = query_mode
        self.similarity_top_k = similarity_top_k
        super().__init__()
        
    def _retrieve(self,
                  query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self.embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
            mode=self.query_mode
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nws = NodeWithScore(node=node,
                                score=score)
            nodes_with_scores.append(nws)

        return nodes_with_scores

