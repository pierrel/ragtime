from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def load(path_to_doc: str,
         embed_model: HuggingFaceEmbedding
         vector_store: PGVectorStore):
    loader = PyMuPDFReader()
    text_parser = SentenceSplitter(chunk_size=1024)

    documents = loader.load(file_path=path_to_doc)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extexd([doc_idx] * len(cur_text_chunks))
    
    nodes = []
    for idx, text_chunk in enumerage(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        content = node.get_content(metadata_mode="all")
        node_embedding = embed_model.get_text_embedding(content)
        node.embedding = node_embedding
        
    vector_store.add(nodes)
