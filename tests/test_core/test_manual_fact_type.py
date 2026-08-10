"""End-to-end contracts for caller-selected memory fact classifications."""

from __future__ import annotations


def test_canonical_store_preserves_explicit_fact_type(engine_with_mock_deps) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    engine = engine_with_mock_deps
    result = canonical_store(
        engine,
        "I strongly prefer PostgreSQL for durable transactional application data.",
        source_type="test",
        trusted_actor_id=local_trusted_actor_id("test"),
        metadata={"fact_type": "opinion"},
        require_complete=True,
        return_receipt=True,
    )

    facts = engine._db.get_facts_by_ids(
        list(result.final_fact_ids), engine.profile_id,
    )
    assert facts
    assert {fact.fact_type.value for fact in facts} == {"opinion"}
