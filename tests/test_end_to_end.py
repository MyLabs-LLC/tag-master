"""End-to-end smoke test on a synthetic corpus.

Runs the real training functions -- not simplified stand-ins -- so the wiring
between feature spaces, both tier heads, calibration, the joint decoder, and the
saved bundle is all exercised without downloading the 307 MB dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from tagmaster.bundle import ModelBundle
from tagmaster.config import load_config
from tagmaster.data import Splits, assert_no_uid_leakage
from tagmaster.evaluate import evaluate_predictions
from tagmaster.features import EmbeddingStore
from tagmaster.headings import HeadingMatcher
from tagmaster.hierarchy import JointDecoder, tune_lambda
from tagmaster.predict import Predictor
from tagmaster.taxonomy import OTHER, TagVocabulary, build_taxonomy
from tagmaster.train import train_tier1, train_tier2, tier2_log_probabilities

# Each entry is (domain, tier-1 class, sub-tag, distinctive body vocabulary).
RECIPES = [
    ("Healthcare", "healthcare", "discharge summary", "patient discharged ward attending nurse recovery vitals"),
    ("Health", "healthcare", "prescription", "pharmacy dosage milligrams refill tablets pharmacist dispense"),
    ("Finance", "finance", "bank statement", "account balance deposits withdrawals interest accrued ledger"),
    ("Banking", "finance", "loan application", "borrower principal repayment collateral amortisation lender"),
    ("Government", "government", "voter registration", "precinct ballot electoral roll constituency polling"),
    ("Elections", "government", "passport", "citizenship travel document issuing authority visa consular"),
    ("Chemicals", OTHER, "safety data sheet", "solvent corrosive ventilation goggles flammable hazard"),
    ("Agriculture", OTHER, "harvest report", "acreage silage irrigation tractor yield hectare topsoil"),
]

FILLERS = [
    "reference number {n} recorded on file",
    "prepared by the regional office in quarter {n}",
    "revision {n} supersedes all previous versions",
    "contact the administrator for clarification item {n}",
]


def _make_document(recipe: tuple[str, str, str, str], n: int) -> str:
    _, _, tag, vocabulary = recipe
    filler = FILLERS[n % len(FILLERS)].format(n=n)
    words = vocabulary.split()
    body = " ".join(words[: 3 + (n % (len(words) - 2))])
    return f"**{tag.title()}**\n\n{body}\n\n{filler}\n\n{vocabulary}"


def _build_splits(per_recipe: int = 40) -> tuple[Splits, TagVocabulary]:
    """Synthetic corpus mirroring the real one's structure, including uid pairing."""
    rows: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    counter = 0
    for recipe in RECIPES:
        domain, top, tag, _ = recipe
        for i in range(per_recipe):
            counter += 1
            uid = f"u{counter:05d}"
            split = "train" if i < per_recipe * 0.6 else ("dev" if i < per_recipe * 0.8 else "test")
            # Two rows per uid, one per locale, exactly as the real dataset.
            for locale in ("us", "intl"):
                rows[split].append(
                    {
                        "uid": uid,
                        "domain": domain,
                        "top": top,
                        "tag": tag,
                        "description": f"A {tag} document from the {domain} domain.",
                        "document_format": "structured" if i % 2 else "unstructured",
                        "locale": locale,
                        "text": _make_document(recipe, i) + (" locale intl" if locale == "intl" else ""),
                    }
                )

    from tests.conftest import make_frame

    splits = Splits(**{name: make_frame(rs) for name, rs in rows.items()})
    tags_by_top: dict[str, list[str]] = {}
    for _, top, tag, _ in RECIPES:
        if top != OTHER:
            tags_by_top.setdefault(top, []).append(tag)
    return splits, TagVocabulary(tags_by_top=tags_by_top, aliases={})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    # Redirect artifacts to a temporary directory, and undo it afterwards so the
    # variable does not leak into the rest of the session.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("TAGMASTER_ARTIFACTS", str(tmp_path_factory.mktemp("artifacts")))
    cfg = load_config()
    cfg.paths.ensure()

    # The synthetic corpus is small, so relax the document-frequency floors that
    # are tuned for 140k real rows.
    for profile in ("tier1", "tier2"):
        cfg.raw["features"][profile]["word_min_df"] = 1
        cfg.raw["features"][profile]["use_embeddings"] = False
    cfg.raw["features"]["tier2"]["char_min_df"] = 1
    cfg.raw["tier1"]["C_grid"] = [1.0]
    cfg.raw["tier2"]["C_grid"] = [1.0]

    taxonomy = build_taxonomy(cfg)
    splits, vocab = _build_splits()
    assert_no_uid_leakage(splits)

    store = EmbeddingStore(None)
    tier1, tier1_space, _ = train_tier1(cfg, taxonomy, splits, store)
    matcher = HeadingMatcher.fit(splits.train.text, vocab, heading_window=400)
    tier2, tier2_space, _ = train_tier2(cfg, taxonomy, splits, vocab, store, matcher)

    decoder = JointDecoder(
        top_labels=list(taxonomy.top_labels),
        industries=list(taxonomy.industries),
        tags_by_top=vocab.tags_by_top,
    )
    Xdv = tier1_space.transform_sparse(splits.dev.text)
    t1_lp = tier1.log_proba(Xdv)
    t2_lp = tier2_log_probabilities(tier2, tier2_space, matcher, splits.dev, None)
    gold_top = splits.dev.top.astype(str)
    gold_tag = np.array(
        [t if s != OTHER else None for s, t in zip(gold_top, splits.dev.tag.tolist())],
        dtype=object,
    )
    tune_lambda(decoder, t1_lp, t2_lp, gold_top, gold_tag, [0.0, 0.3, 1.0])

    bundle = ModelBundle(
        taxonomy=taxonomy,
        vocab=vocab,
        tier1_space=tier1_space,
        tier2_space=tier2_space,
        tier1=tier1,
        tier2=tier2,
        matcher=matcher,
        decoder=decoder,
        metadata={"synthetic": True},
    )
    bundle.save(cfg)
    yield cfg, bundle, splits
    monkeypatch.undo()


def test_learns_the_synthetic_signal(trained):
    cfg, bundle, splits = trained
    store = EmbeddingStore(None)
    X1 = bundle.tier1_space.transform_sparse(splits.test.text)
    t1_lp = bundle.tier1.log_proba(X1)
    t2_lp = tier2_log_probabilities(
        bundle.tier2, bundle.tier2_space, bundle.matcher, splits.test, None
    )
    pred = bundle.decoder.decode(t1_lp, t2_lp)
    report = evaluate_predictions(splits.test, pred, bundle)

    # The synthetic classes are trivially separable, so anything less than near
    # perfect means the pipeline is mis-wired rather than under-fitted.
    assert report["headline"]["tier1_accuracy"] > 0.95
    assert report["headline"]["end_to_end_exact_match"] > 0.95


def test_report_structure_is_complete(trained):
    cfg, bundle, splits = trained
    X1 = bundle.tier1_space.transform_sparse(splits.test.text)
    t2_lp = tier2_log_probabilities(
        bundle.tier2, bundle.tier2_space, bundle.matcher, splits.test, None
    )
    pred = bundle.decoder.decode(bundle.tier1.log_proba(X1), t2_lp)
    report = evaluate_predictions(splits.test, pred, bundle)

    for key in (
        "tier1_accuracy",
        "tier1_macro_f1",
        "tier2_conditional_accuracy",
        "end_to_end_exact_match",
    ):
        assert key in report["headline"]
    assert set(report["per_industry"]) == set(bundle.taxonomy.industries)
    assert set(report["by_locale"]) == {"us", "intl"}
    assert set(report["by_format"]) == {"structured", "unstructured"}
    assert report["coverage"]["tier1"]
    # Coverage curves must be ordered and end at full coverage.
    coverages = [p["coverage"] for p in report["coverage"]["tier1"]]
    assert coverages == sorted(coverages)
    assert coverages[-1] == 1.0


def test_bundle_round_trips_and_predicts(trained):
    cfg, _, _ = trained
    predictor = Predictor.load(cfg)

    result = predictor.predict(
        "**Voter Registration**\n\nprecinct ballot electoral roll constituency polling",
        top_k=3,
    )
    assert result["industry"] == "government"
    assert result["sub_tag"] == "voter registration"
    assert 0.0 <= result["industry_confidence"] <= 1.0
    assert len(result["alternatives"]) == 3
    assert set(result["industry_probabilities"]) == set(cfg.raw["industries"]) | {OTHER}


def test_out_of_domain_document_is_labelled_other(trained):
    cfg, _, _ = trained
    predictor = Predictor.load(cfg)
    result = predictor.predict(
        "**Safety Data Sheet**\n\nsolvent corrosive ventilation goggles flammable hazard"
    )
    assert result["industry"] == OTHER
    assert result["sub_tag"] is None


def test_batch_prediction_matches_single(trained):
    cfg, _, _ = trained
    predictor = Predictor.load(cfg)
    docs = [
        "**Prescription**\n\npharmacy dosage milligrams refill tablets",
        "**Loan Application**\n\nborrower principal repayment collateral lender",
    ]
    batch = predictor.predict_batch(docs)
    assert [b["industry"] for b in batch] == ["healthcare", "finance"]
    assert batch[0] == predictor.predict(docs[0])
