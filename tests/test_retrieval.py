import numpy as np
import pytest

from ml.retrieval.faiss_store import ReferenceIndex, ReferenceMatch, ReferenceRecord, summarize_matches


def _record(i: int, class_name: str) -> ReferenceRecord:
    return ReferenceRecord(
        f"REF{i:04d}", i, f"img-{i}", class_name, "type", "Synthetic", "test", f"refs/{i}.png"
    )


def _unit(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def _index() -> tuple[ReferenceIndex, np.ndarray]:
    rng = np.random.default_rng(0)
    vectors = _unit(rng.normal(size=(6, 8)))
    records = [_record(i, "a" if i < 3 else "b") for i in range(6)]
    return ReferenceIndex.build(vectors, records, "index-test", "fp"), vectors


def test_search_returns_exact_self_match_first_with_cosine_similarity():
    index, vectors = _index()
    matches = index.search(vectors[4], k=3)[0]
    assert matches[0].reference.reference_id == "REF0004"
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert [m.rank for m in matches] == [1, 2, 3]
    assert [m.similarity for m in matches] == sorted((m.similarity for m in matches), reverse=True)


def test_search_caps_k_at_library_size():
    index, vectors = _index()
    assert len(index.search(vectors[:2], k=50)[1]) == 6


def _match(rank: int, similarity: float, class_name: str) -> ReferenceMatch:
    return ReferenceMatch(rank, similarity, _record(rank, class_name))


def test_summarize_uses_similarity_weighted_vote():
    matches = (_match(1, 0.9, "a"), _match(2, 0.5, "b"), _match(3, 0.5, "b"))
    summary = summarize_matches(matches)
    assert summary.retrieved_class == "b"
    assert summary.top_similarity == 0.9
    shares = {item.class_name: item.weighted_share for item in summary.class_support}
    assert shares["b"] == pytest.approx(1.0 / 1.9)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_summarize_breaks_ties_towards_best_single_match():
    matches = (_match(1, 0.8, "b"), _match(2, 0.4, "a"), _match(3, 0.4, "a"))
    assert summarize_matches(matches).retrieved_class == "b"


def test_save_and_load_round_trip(tmp_path):
    index, vectors = _index()
    index.save(tmp_path, {"created_at": "now"})
    loaded = ReferenceIndex.load(
        tmp_path / ReferenceIndex.INDEX_FILE, tmp_path / ReferenceIndex.METADATA_FILE
    )
    assert (loaded.version, loaded.fingerprint) == ("index-test", "fp")
    assert loaded.records == index.records
    original = [m.reference.reference_id for m in index.search(vectors[0], 6)[0]]
    assert [m.reference.reference_id for m in loaded.search(vectors[0], 6)[0]] == original


def test_rejects_metadata_size_mismatch():
    _, vectors = _index()
    with pytest.raises(ValueError, match="does not match"):
        ReferenceIndex.build(vectors, [_record(0, "a")], "v", "fp")
