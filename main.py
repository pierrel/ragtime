from auth import hf_token
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2

HFTOKEN = hf_token()
DBNAME = "ragtime-staging"
DBUSER="pierre"

def create_embeddings_table(connection):
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

vector_store = PGVectorStore.from_params(
    database=DBNAME,
    user=DBUSER,
    table_name="llama2_paper",
    embed_dim=384,
)
