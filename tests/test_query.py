"""Tests for metadata filtering and stats evidence injection.

Uses real Milvus, real SQLite metadata DB, real alias map — the same
path the CLI takes.  Skips the LLM/reranker (no NIM calls).

Run:  uv run pytest tests/test_query.py -v
"""

from __future__ import annotations

import pytest

from nim_opensansad import config
from nim_opensansad.metadata import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_DB_PATH,
    load_alias_map,
)
from nim_opensansad.query import build_milvus_filter
from nim_opensansad.stats import build_evidence_packet


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def alias_map():
    if not DEFAULT_ALIAS_PATH.exists():
        pytest.skip("Alias map not found — run 'opensansad build-aliases' first")
    return load_alias_map(DEFAULT_ALIAS_PATH)


@pytest.fixture(scope="module")
def milvus():
    from pymilvus import MilvusClient
    client = MilvusClient(uri=config.MILVUS_URI)
    if not client.has_collection(config.COLLECTION_NAME):
        pytest.skip("Milvus collection not available")
    return client


def _query_milvus(client, expr: str, limit: int = 20):
    return client.query(
        config.COLLECTION_NAME,
        filter=expr,
        output_fields=["members", "ministry", "qa_id"],
        limit=limit,
    )


# ── Metadata filtering ──────────────────────────────────────────────────────


class TestMetadataFiltering:

    def test_mp_filter_with_aliases(self, milvus, alias_map):
        """Filter for Rahul Gandhi using alias expansion — every returned
        chunk must contain at least one of his name variants."""
        mp = "Shri Rahul Gandhi"
        aliases = alias_map[mp]
        expr = build_milvus_filter(mp=mp, mp_aliases=aliases)
        results = _query_milvus(milvus, expr)

        assert len(results) > 0, f"No chunks returned for {mp}"
        for r in results:
            members = r["members"]
            assert any(a in members for a in aliases), (
                f"Chunk {r['qa_id']} members='{members}' "
                f"doesn't match any alias: {aliases}"
            )

    def test_ministry_filter(self, milvus):
        """Exact ministry filter — every chunk must match."""
        ministry = "DEFENCE"
        expr = build_milvus_filter(ministry=ministry)
        results = _query_milvus(milvus, expr)

        assert len(results) > 0, f"No chunks for ministry: {ministry}"
        for r in results:
            assert r["ministry"] == ministry

    def test_combined_mp_and_ministry(self, milvus, alias_map):
        """Both filters applied — results must satisfy both."""
        mp = "Shri Rahul Gandhi"
        ministry = "HEALTH AND FAMILY WELFARE"
        aliases = alias_map[mp]
        expr = build_milvus_filter(mp=mp, ministry=ministry, mp_aliases=aliases)
        results = _query_milvus(milvus, expr)

        # May be 0 if no overlap, but whatever comes back must match both
        for r in results:
            assert any(a in r["members"] for a in aliases)
            assert r["ministry"] == ministry


# ── Evidence injection ───────────────────────────────────────────────────────


class TestEvidenceInjection:

    def test_evidence_built_from_real_db(self):
        """build_evidence_packet returns real stats from the SQLite DB."""
        if not DEFAULT_DB_PATH.exists():
            pytest.skip("Metadata DB not found")
        evidence = build_evidence_packet(mp_name="Shri Rahul Gandhi")
        assert evidence is not None
        assert "Rahul Gandhi" in evidence
        assert "Total questions" in evidence

    def test_evidence_in_prompt_not_query(self):
        """The engine should bake evidence into the synthesizer prompt
        template, leaving the user query untouched."""
        from nim_opensansad.query import STATS_QA_TMPL

        evidence = build_evidence_packet(mp_name="Shri Rahul Gandhi")
        assert evidence is not None

        # This is what build_query_engine does internally
        rendered = STATS_QA_TMPL.replace("{evidence_str}", evidence)

        # Evidence is in the template
        assert "Rahul Gandhi" in rendered
        # The two LlamaIndex placeholders survive — they get filled at
        # query time with the retrieved context and the user's raw query
        assert "{context_str}" in rendered
        assert "{query_str}" in rendered


# ── Integration: full retriever pipeline ─────────────────────────────────────


class TestRetrieverIntegration:
    """End-to-end through VectorIndexRetriever → MilvusVectorStore,
    using the real embed model and real Milvus.  No LLM or reranker."""

    @pytest.fixture(scope="class")
    def retriever_components(self):
        """Build the shared embed model + vector store index once."""
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.milvus import MilvusVectorStore
        from nim_opensansad.ingest import _get_device

        embed_model = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL,
            device=_get_device(),
            query_instruction="query: ",
            text_instruction="passage: ",
        )
        vector_store = MilvusVectorStore(
            uri=config.MILVUS_URI,
            collection_name=config.COLLECTION_NAME,
            dim=config.EMBED_DIM,
            overwrite=False,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model,
        )
        return index

    def _retrieve(self, index, query: str, mp=None, ministry=None, mp_aliases=None):
        from llama_index.core.retrievers import VectorIndexRetriever

        milvus_filter = build_milvus_filter(mp=mp, ministry=ministry, mp_aliases=mp_aliases)
        vs_kwargs = {"string_expr": milvus_filter} if milvus_filter else {}

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=config.TOP_K,
            vector_store_kwargs=vs_kwargs,
        )
        return retriever.retrieve(query)

    def test_mp_filter_through_retriever(self, retriever_components, alias_map):
        """Retriever with MP filter — every node must belong to that MP."""
        mp = "Shri Rahul Gandhi"
        aliases = alias_map[mp]
        nodes = self._retrieve(
            retriever_components,
            "What issues has Rahul Gandhi raised?",
            mp=mp, mp_aliases=aliases,
        )
        assert len(nodes) > 0, "Retriever returned no nodes"
        for node in nodes:
            members = node.metadata.get("members", "")
            assert any(a in members for a in aliases), (
                f"Node {node.metadata.get('qa_id')} members='{members}' "
                f"not matching aliases {aliases}"
            )

    def test_ministry_filter_through_retriever(self, retriever_components):
        """Retriever with ministry filter — every node must match."""
        ministry = "DEFENCE"
        nodes = self._retrieve(
            retriever_components,
            "What are recent defence related questions?",
            ministry=ministry,
        )
        assert len(nodes) > 0, "Retriever returned no nodes"
        for node in nodes:
            assert node.metadata.get("ministry") == ministry

    def test_unfiltered_retriever(self, retriever_components):
        """Without filters, retriever should still return results."""
        nodes = self._retrieve(
            retriever_components,
            "Has there been a discussion on remdesivir?",
        )
        assert len(nodes) > 0, "Retriever returned no nodes"
