"""
Query pipeline: retrieve → rerank → synthesize.

Pipeline:
  User query
    → HuggingFaceEmbedding (same local model used at ingest time)
    → MilvusVectorStore (top_k candidates, with optional metadata filters)
    → NVIDIARerank (reranker NIM, narrows to rerank_top_n)
    → NVIDIA LLM (synthesis with cited sources + optional stats evidence)
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.vector_stores.milvus import MilvusVectorStore

from . import config
from .ingest import _get_device


# ── Prompt templates ─────────────────────────────────────────────────────────

DEFAULT_QA_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer:"
)

# When stats evidence is available, it is injected as a separate section
# so the LLM sees it alongside the retrieved chunks — not in the query.
STATS_QA_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "The following aggregate statistics from the parliamentary database "
    "are also available to help you give a comprehensive answer.\n"
    "---------------------\n"
    "{evidence_str}\n"
    "---------------------\n"
    "Given the context information, statistics, and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer:"
)


# ── Milvus filter builder ───────────────────────────────────────────────────

def build_milvus_filter(
    mp: str | None = None,
    ministry: str | None = None,
    mp_aliases: list[str] | None = None,
) -> str:
    """Build a native Milvus boolean expression for metadata filtering.

    Uses ``like "%name%"`` for the members field (comma-separated list)
    and exact ``==`` for ministry.

    Args:
        mp: Canonical MP name (used only if mp_aliases is not provided).
        ministry: Ministry name — exact match.
        mp_aliases: All name variants for the MP (canonical + historical).
            When provided, generates an OR across all variants so that
            chunks from any Lok Sabha are matched.
    """
    conditions: list[str] = []

    names = mp_aliases if mp_aliases else ([mp] if mp else [])
    if names:
        like_clauses = []
        for name in names:
            safe = name.replace('"', '\\"')
            like_clauses.append(f'members like "%{safe}%"')
        if len(like_clauses) == 1:
            conditions.append(like_clauses[0])
        else:
            conditions.append("(" + " or ".join(like_clauses) + ")")

    if ministry:
        safe_min = ministry.replace('"', '\\"')
        conditions.append(f'ministry == "{safe_min}"')
    return " and ".join(conditions)


# ── Engine builder ───────────────────────────────────────────────────────────

def build_query_engine(
    mp: str | None = None,
    ministry: str | None = None,
    evidence: str | None = None,
    mp_aliases: list[str] | None = None,
) -> RetrieverQueryEngine:
    """
    Assemble the full retrieval + rerank + synthesis pipeline.

    Args:
        mp: Optional MP name — filters Milvus retrieval to chunks where
            ``members`` contains this name.
        ministry: Optional ministry name — filters Milvus retrieval to
            chunks from this ministry.
        evidence: Optional stats evidence text to inject into the LLM
            synthesis prompt (alongside retrieved context, NOT in the query).
        mp_aliases: All name variants for the MP. When provided, the Milvus
            filter will OR across all variants.
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

    # Metadata filtering via native Milvus expression
    milvus_filter = build_milvus_filter(mp=mp, ministry=ministry, mp_aliases=mp_aliases)
    vs_kwargs: dict = {}
    if milvus_filter:
        vs_kwargs["string_expr"] = milvus_filter

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=config.TOP_K,
        vector_store_kwargs=vs_kwargs,
    )

    # Build the synthesis prompt — with or without stats evidence
    if evidence:
        qa_prompt = PromptTemplate(
            STATS_QA_TMPL.replace("{evidence_str}", evidence)
        )
    else:
        qa_prompt = PromptTemplate(DEFAULT_QA_TMPL)

    synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[reranker],
    )
