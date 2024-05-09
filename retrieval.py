from llama_index.core.vector_stores import VectorStoreQuery

def retrieve(query:str,
             embed_model:HuggingFaceEmbedding,
             vector_store:PGVectorStore) -> str:
    query_embedding = embed_model.get_query_embedding(query)
    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=2,
        mode="default"
    )
    query_result = vector_store.query(vector_store_query)
    return query_result.nodes[0].get_content()
# continue from https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/#3-parse-result-into-a-set-of-nodes
