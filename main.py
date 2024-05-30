from auth import hf_token
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine
from sqlalchemy import make_url
import psycopg2
from ingest import VectorDBLoader
from retrieval import VectorDBRetriever

HFTOKEN = hf_token()
DBNAME = "ragtime-staging"
DBUSER="pierre"

def create_embeddings_table(connection):
    """Currently not being used, but left here as an example"""
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE embeddings (id serial PRIMARY KEY, vector vector(512));")
    connection.commit();

conn = psycopg2.connect(dbname=DBNAME, user=DBUSER)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en", token=HFTOKEN)
embed_model.get_text_embedding
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

connection_string = 'postgresql://pierre@localhost:5432/ragtime-staging'
connection_url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=DBNAME,
    user=connection_url.username,
    host=connection_url.host,
    port=connection_url.port,
    table_name="llama2_paperr",
    embed_dim=384,
)
vector_store.client # initialize things

loader = VectorDBLoader(embed_model, vector_store)
retriever = VectorDBRetriever(vector_store,
                              embed_model,
                              query_mode="default",
                              similarity_top_k=2)
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

# Query!
response = query_engine.query("How does Llama 2 perform compared to other open-sourced models?")
