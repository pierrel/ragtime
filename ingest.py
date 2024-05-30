from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import TextNode

import pdb

class VectorDBLoader:
    def __init__(self,
                 embed_model: HuggingFaceEmbedding,
                 vector_store: PGVectorStore):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.loader = PyMuPDFReader()
        self.text_parser = SentenceSplitter(chunk_size=1024)

    def load(self, path_to_doc: str):
        documents = self.loader.load(file_path=path_to_doc)
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = self.text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))
            
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        for node in nodes:
            content = node.get_content(metadata_mode="all")
            node_embedding = self.embed_model.get_text_embedding(content)
            node.embedding = node_embedding
        self.vector_store.add(nodes)
