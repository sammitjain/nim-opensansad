"""
Query pipeline: retrieve → rerank → synthesize.

Pipeline:
  User query
    → HuggingFaceEmbedding (same local model used at ingest time)
    → MilvusVectorStore (top_k candidates)
    → NVIDIARerank (reranker NIM, narrows to rerank_top_n)
    → NVIDIA LLM (synthesis with cited sources)
"""

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.vector_stores.milvus import MilvusVectorStore

from . import config
from .ingest import _get_device


def build_query_engine() -> RetrieverQueryEngine:
    """
    Assemble the full retrieval + rerank + synthesis pipeline.
    Call once and reuse the returned engine for multiple queries.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL,
        device=_get_device(),
        query_instruction="query: ",
        text_instruction="passage: ",
    )
    llm = NVIDIA(
        model=config.LLM_MODEL,
        api_key=config.NVIDIA_API_KEY,
    )
    reranker = NVIDIARerank(
        model=config.RERANK_MODEL,
        api_key=config.NVIDIA_API_KEY,
        top_n=config.RERANK_TOP_N,
    )

    vector_store = MilvusVectorStore(
        uri=config.MILVUS_URI,
        collection_name=config.COLLECTION_NAME,
        dim=config.EMBED_DIM,
        overwrite=False,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=config.TOP_K,
    )
    synthesizer = get_response_synthesizer(llm=llm)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[reranker],
    )
