from __future__ import annotations

import pytest

from tagmaster.config import Config
from tagmaster.taxonomy import (
    OTHER,
    TagVocabulary,
    TaxonomyError,
    apply_aliases,
    build_taxonomy,
    normalize_tag,
    tag_tokens,
)


def test_core_mapping_collapses_adjacent_domains(taxonomy):
    # The whole point of the core mapping: the shipped splits name the same
    # industry differently, so Health and Healthcare must land together.
    assert taxonomy.top_for_domain("Healthcare") == "healthcare"
    assert taxonomy.top_for_domain("Health") == "healthcare"
    assert taxonomy.top_for_domain("Healthcare Providers") == "healthcare"
    assert taxonomy.top_for_domain("Finance") == "finance"
    assert taxonomy.top_for_domain("Banking") == "finance"
    assert taxonomy.top_for_domain("Mortgage") == "finance"
    assert taxonomy.top_for_domain("Government") == "government"
    assert taxonomy.top_for_domain("Elections") == "government"
    assert taxonomy.top_for_domain("Public Safety") == "government"


def test_unmapped_domains_fall_to_other(taxonomy):
    for domain in ("Chemicals", "Agriculture", "Legal", "Insurance", "Pharmaceuticals"):
        assert taxonomy.top_for_domain(domain) == OTHER


def test_top_labels_put_other_last(taxonomy):
    assert taxonomy.top_labels[-1] == OTHER
    assert len(taxonomy.top_labels) == 4


def test_hard_negatives_are_other_domains(taxonomy):
    assert taxonomy.is_hard_negative("Legal")
    assert taxonomy.is_hard_negative("Pharmaceuticals")
    assert not taxonomy.is_hard_negative("Healthcare")
    assert not taxonomy.is_hard_negative("Chemicals")


def test_unknown_domain_raises_rather_than_silently_becoming_other(taxonomy):
    with pytest.raises(TaxonomyError, match="missing from configs"):
        taxonomy.validate_domains({"Healthcare", "Cryptozoology"})


def test_known_domains_cover_the_dataset(taxonomy):
    # 58 distinct domains across the two shipped splits.
    assert len(taxonomy.known_domains) == 58


def test_industry_claimed_twice_is_rejected():
    raw = {
        "industries": {"healthcare": ["Health"], "finance": ["Health"]},
        "known_domains": ["Health"],
        "split": {"seed": 1},
    }
    with pytest.raises(TaxonomyError, match="claimed by both"):
        build_taxonomy(Config(raw=raw, path=None))


def test_hard_negative_overlapping_industry_is_rejected():
    raw = {
        "industries": {"finance": ["Banking"]},
        "known_domains": ["Banking"],
        "hard_negative_domains": {"finance": ["Banking"]},
        "split": {"seed": 1},
    }
    with pytest.raises(TaxonomyError, match="mapped to an industry"):
        build_taxonomy(Config(raw=raw, path=None))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Admission Checklist", "admission checklist"),
        ("admission checklist", "admission checklist"),
        ("Medical Records", "medical record"),
        ("medical record", "medical record"),
        ("Tax Return Form (e.g., IRS 1040)", "tax return form e g irs 1040"),
        ("  Voter   Registration  ", "voter registration"),
        ("Address", "address"),
        ("", ""),
    ],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


def test_normalize_tag_preserves_double_s():
    # Stripping the trailing "s" from "address" would corrupt the tag.
    assert normalize_tag("Address Verification") == "address verification"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Medical Diagnosis", "medical diagnosis"),
        ("Census Response Form", "census response form"),
        ("Fund Prospectus", "fund prospectus"),
        ("Market Analysis Report", "market analysis report"),
        ("Arthritis Management Plan", "arthritis management plan"),
        ("Affidavit of Marital Status", "affidavit of marital status"),
    ],
)
def test_normalize_tag_leaves_false_plurals_intact(raw, expected):
    # "diagnosis" is not the plural of "diagnosi".
    assert normalize_tag(raw) == expected


def test_normalize_tag_still_folds_real_plurals():
    assert normalize_tag("Medical Records") == normalize_tag("Medical Record")
    assert normalize_tag("Claim Forms") == "claim form"


def test_tag_tokens_drops_only_structural_words():
    assert tag_tokens("affidavit of no unpaid fee") == frozenset(
        {"affidavit", "no", "unpaid", "fee"}
    )


def test_vocabulary_indexes_per_industry():
    vocab = TagVocabulary(
        tags_by_top={"healthcare": ["prescription", "discharge summary"], "finance": ["loan"]},
        aliases={},
    )
    assert vocab.total == 3
    assert vocab.tags("healthcare") == ["discharge summary", "prescription"]
    assert vocab.index("healthcare", "prescription") == 1
    assert vocab.index("finance", "prescription") is None


def test_aliases_are_scoped_to_one_industry():
    vocab = TagVocabulary(
        tags_by_top={"finance": ["tax return", "tax form"], "government": ["tax return"]},
        aliases={},
    )
    coarse = apply_aliases(vocab, {"finance\ttax form": "tax return"})
    assert coarse.tags("finance") == ["tax return"]
    # The government tag of the same name is untouched.
    assert coarse.tags("government") == ["tax return"]
    assert coarse.canonical("finance", "tax form") == "tax return"
    assert coarse.canonical("government", "tax form") == "tax form"
