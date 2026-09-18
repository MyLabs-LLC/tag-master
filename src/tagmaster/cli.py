"""Command line entry point: `tagmaster prepare | train | evaluate | predict`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config

app = typer.Typer(add_completion=False, help="Hierarchical industry and sub-tag tagger.")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to taxonomy.yaml.")


@app.command()
def prepare(
    config: Optional[Path] = ConfigOpt,
    rebuild: bool = typer.Option(False, help="Re-download and re-split instead of using caches."),
) -> None:
    """Download the dataset, build uid-grouped splits, derive the tier-2 labels."""
    from .prepare import run_prepare

    cfg = load_config(config)
    report = run_prepare(cfg, rebuild=rebuild)
    counts = report["counts"]["splits"]
    typer.echo(
        "rows  train={train} dev={dev} test={test}".format(
            train=counts["train"]["rows"], dev=counts["dev"]["rows"], test=counts["test"]["rows"]
        )
    )
    for top, stats in report["tier2_label_space"]["per_industry"].items():
        typer.echo(f"  {top:<11} {stats['tags']:>4} sub-tags  {stats['train_uids']:>6} train uids")
    typer.echo(f"  total sub-tags: {report['tier2_label_space']['total_tags']}")
    typer.echo(f"report -> {cfg.paths.reports / 'prepare.json'}")


@app.command("warm-cache")
def warm_cache(config: Optional[Path] = ConfigOpt) -> None:
    """Pre-compute and cache document embeddings for every split.

    Optional but worthwhile: embedding runs at roughly 58 documents/second on
    four CPU threads, so doing it once up front keeps later training and tuning
    runs fast.
    """
    from .data import get_splits
    from .embed import build_embedder
    from .taxonomy import build_taxonomy

    cfg = load_config(config)
    embedder = build_embedder(cfg)
    if embedder is None:
        typer.echo("embeddings are disabled in the config; nothing to warm")
        return
    splits = get_splits(cfg, build_taxonomy(cfg))
    for name, frame in splits.items():
        typer.echo(f"{name}: {len(frame)} rows")
        embedder.encode_cached([(t or "")[:2000] for t in frame.text.tolist()], name)
    typer.echo(f"cache -> {cfg.paths.cache}")


@app.command()
def train(
    config: Optional[Path] = ConfigOpt,
    skip_embeddings: bool = typer.Option(False, help="Train on TF-IDF and heading features only."),
) -> None:
    """Fit the tier-1 head, the tier-2 heads, and tune the joint decoder."""
    from .train import run_train

    cfg = load_config(config)
    summary = run_train(cfg, use_embeddings=not skip_embeddings)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def evaluate(
    config: Optional[Path] = ConfigOpt,
    split: str = typer.Option("test", help="Which split to score: dev or test."),
) -> None:
    """Score the trained model and write the full metrics report."""
    from .evaluate import run_evaluate

    cfg = load_config(config)
    report = run_evaluate(cfg, split=split)
    typer.echo(json.dumps(report["headline"], indent=2))
    typer.echo(f"report -> {cfg.paths.reports / f'evaluation_{split}.json'}")


@app.command()
def predict(
    config: Optional[Path] = ConfigOpt,
    text: Optional[str] = typer.Option(None, help="Document text to classify."),
    file: Optional[Path] = typer.Option(None, help="Read the document from this file."),
    top_k: int = typer.Option(3, help="How many candidate (industry, sub-tag) pairs to show."),
) -> None:
    """Classify a single document."""
    from .predict import Predictor

    if text is None and file is None:
        raise typer.BadParameter("pass --text or --file")
    document = text if text is not None else Path(file).read_text(encoding="utf-8")

    cfg = load_config(config)
    predictor = Predictor.load(cfg)
    typer.echo(json.dumps(predictor.predict(document, top_k=top_k), indent=2))


if __name__ == "__main__":
    app()
