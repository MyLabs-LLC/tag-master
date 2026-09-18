from __future__ import annotations

import numpy as np

from tagmaster.headings import HeadingMatcher, extract_headings, tokenize
from tagmaster.taxonomy import TagVocabulary, normalize_tag

STRUCTURED = (
    "**Physical Therapy Plan**\n\n**Patient Information**\n\n"
    "First Name: Oliver\n\nMedical Record Number: 0012345678\n"
)
MARKDOWN = "### Compliance Report\n\n**Executive Summary**\n\nHarvest Financial Group is committed"
FENCED = "```\nEmployment Verification Form\n\nEmployee Name: Lawrence\n"
TABLE = (
    "**Healthcare Provider Information**\n\n"
    "| Field | Details |\n|-------|---------|\n| Provider ID | BIO-5729183645 |\n"
)
LETTER = (
    "To Whom It May Concern,\n\nThis letter serves as an Income Verification for "
    "Stanley Richardson, Employee ID: 003782."
)


def test_extracts_bold_title():
    assert extract_headings(STRUCTURED)[0] == "Physical Therapy Plan"


def test_extracts_markdown_heading():
    assert extract_headings(MARKDOWN)[0] == "Compliance Report"


def test_skips_code_fence_and_finds_the_title_below():
    headings = extract_headings(FENCED)
    assert "```" not in headings
    assert "Employment Verification Form" in headings


def test_skips_table_rows_and_rule_lines():
    headings = extract_headings(TABLE)
    assert headings[0] == "Healthcare Provider Information"
    assert not any(h.startswith("|") or set(h) <= set("-| ") for h in headings)


def test_skips_salutation_but_keeps_following_content():
    headings = extract_headings(LETTER)
    assert not any(h.lower().startswith("to whom") for h in headings)
    assert any("Income Verification" in h for h in headings)


def test_empty_text_is_safe():
    assert extract_headings("") == []
    assert extract_headings(None) == []


def test_tokenize_drops_digits_and_single_characters():
    assert tokenize("Provider ID BIO-5729183645 a") == ["provider", "id", "bio"]


# Background documents so inverse document frequency is meaningful; with a
# two-document corpus every token has df == n and IDF collapses to zero.
BACKGROUND = [
    "Quarterly energy production summary for the northern region and its plants",
    "Agricultural yield survey covering maize soy and wheat across twelve counties",
    "Chemical safety data sheet listing handling storage and disposal procedures",
    "Manufacturing line throughput report with downtime causes and shift totals",
    "Retail inventory reconciliation for seasonal apparel and footwear lines",
    "Logistics shipment manifest describing freight weights and delivery windows",
]


def _matcher(tags_by_top, corpus):
    vocab = TagVocabulary(tags_by_top=tags_by_top, aliases={})
    return HeadingMatcher.fit(np.array(corpus + BACKGROUND, dtype=object), vocab), vocab


def test_heading_channel_ranks_the_titled_tag_first():
    tags = {"healthcare": ["physical therapy plan", "discharge summary", "prescription"]}
    corpus = [STRUCTURED, MARKDOWN, "prescription for amoxicillin", "discharge summary follows"]
    matcher, vocab = _matcher(tags, corpus)
    scores = matcher.score_document(STRUCTURED, "healthcare")
    best = vocab.tags("healthcare")[int(scores.heading.argmax())]
    assert best == "physical therapy plan"


def test_verbatim_channel_fires_on_mid_sentence_mentions():
    # The unstructured half of the corpus states the tag in prose, not a title.
    tags = {"finance": ["investment agreement", "loan application"]}
    text = "This Brokerage and Investment Agreement is entered into between two parties."
    matcher, vocab = _matcher(tags, [text, "loan application form"])
    scores = matcher.score_document(text, "finance")
    best = vocab.tags("finance")[int(scores.verbatim.argmax())]
    assert best == "investment agreement"
    assert scores.verbatim.max() > 0


def test_verbatim_channel_is_position_weighted():
    tags = {"finance": ["loan application"]}
    early = "Loan Application\n\n" + "filler text " * 40
    late = "filler text " * 40 + "\n\nLoan Application"
    matcher, _ = _matcher(tags, [early, late])
    assert (
        matcher.score_document(early, "finance").verbatim[0]
        > matcher.score_document(late, "finance").verbatim[0]
    )


def test_subset_heading_earns_partial_credit():
    # "Brokerage and Confidentiality Agreement" should still favour
    # `confidentiality agreement` over an unrelated tag.
    tags = {"government": ["confidentiality agreement", "birth certificate"]}
    text = "**Brokerage and Confidentiality Agreement**\n\nThis agreement is entered into on 2023"
    matcher, vocab = _matcher(tags, [text, "birth certificate record"])
    scores = matcher.score_document(text, "government")
    best = vocab.tags("government")[int(scores.heading.argmax())]
    assert best == "confidentiality agreement"


def test_score_matrix_shape_matches_label_space():
    tags = {"finance": ["loan application", "bank statement", "credit report"]}
    texts = np.array([STRUCTURED, MARKDOWN, LETTER], dtype=object)
    matcher, _ = _matcher(tags, texts.tolist())
    scores = matcher.score_matrix(texts, "finance")
    assert scores.heading.shape == (3, 3)
    assert scores.verbatim.shape == (3, 3)


def test_unknown_industry_returns_empty_scores():
    matcher, _ = _matcher({"finance": ["loan application"]}, ["loan application"])
    scores = matcher.score_document(STRUCTURED, "healthcare")
    assert scores.heading.shape == (0,)


def test_verbatim_rate_matches_a_known_corpus():
    tags = {"healthcare": ["physical therapy plan"]}
    matcher, _ = _matcher(tags, [STRUCTURED])
    rate = matcher.verbatim_rate(
        np.array([STRUCTURED, "unrelated text"], dtype=object),
        np.array(["physical therapy plan", "physical therapy plan"], dtype=object),
    )
    assert rate == 0.5


def test_normalize_tag_bridges_heading_and_label():
    assert normalize_tag("Physical Therapy Plan") == "physical therapy plan"
