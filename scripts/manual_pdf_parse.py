"""Parse one digital PDF and inspect InsightFlow's normalized representation."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.models import DocumentSource, NormalizedDocument
from insightflow.rag.parsers import DoclingPdfParser


def parse_args() -> argparse.Namespace:
    """Read manual-parser arguments."""
    parser = argparse.ArgumentParser(
        description="Parse one digital PDF with Docling and inspect its normalized elements.",
    )
    parser.add_argument("pdf", type=Path, help="Path to a digital PDF with selectable text.")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="Corpus root used to derive a stable relative source identifier.",
    )
    parser.add_argument(
        "--source-id",
        help="Explicit stable source identifier; overrides --corpus-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete NormalizedDocument JSON output.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the complete normalized document text after the element summary.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="Maximum characters shown for each element preview (default: 160).",
    )
    return parser.parse_args()


def source_identifier(pdf: Path, corpus_root: Path | None, explicit: str | None) -> str:
    """Choose a stable corpus-relative identity for the source."""
    if explicit:
        return explicit
    if corpus_root is None:
        return pdf.name
    try:
        return pdf.relative_to(corpus_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{pdf} is not inside corpus root {corpus_root}") from exc


def make_source(args: argparse.Namespace) -> DocumentSource:
    """Build the parser input from validated CLI paths."""
    pdf = args.pdf.expanduser().resolve()
    corpus_root = args.corpus_root.expanduser().resolve() if args.corpus_root else None
    identifier = source_identifier(pdf, corpus_root, args.source_id)
    return DocumentSource(
        source_name=pdf.name,
        source_identifier=identifier,
        path=pdf,
        mime_type="application/pdf",
    )


def preview(text: str, limit: int) -> str:
    """Render one compact, single-line element preview."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 1)]}…"


def print_document(document: NormalizedDocument, preview_chars: int) -> None:
    """Print identity, counts, hierarchy, pages, and element previews."""
    counts = Counter(element.element_type for element in document.elements)
    print("\nNormalized document")
    print(f"  source:       {document.source_identifier}")
    print(f"  document_id:  {document.document_id}")
    print(f"  version:      {document.document_version}")
    print(f"  title:        {document.title or '(not detected)'}")
    print(f"  pages:        {document.metadata.get('page_count', '(unknown)')}")
    print(f"  characters:   {len(document.text)}")
    print(f"  elements:     {len(document.elements)}")
    print(f"  element types: {dict(sorted(counts.items()))}")
    print("\nElements in reading order")

    for element in document.elements:
        page = str(element.page_number) if element.page_number is not None else "?"
        heading = " > ".join(element.heading_path) or "(document root)"
        offsets = f"{element.char_start}:{element.char_end}"
        print(
            f"[{element.order:04d}] {element.element_type:<10} "
            f"page={page:<4} chars={offsets:<15} heading={heading}"
        )
        print(f"       {preview(element.text, preview_chars)}")
        if element.metadata:
            print(f"       metadata={element.metadata}")


async def run(args: argparse.Namespace) -> int:
    """Run one real Docling conversion and optionally persist its domain output."""
    try:
        source = make_source(args)
    except ValueError as exc:
        print(f"Invalid input: {exc}")
        return 2

    print(f"Parsing {source.path} with OCR disabled...")
    try:
        document = await DoclingPdfParser().parse(source)
    except DocumentRejectedError as exc:
        print(f"Rejected: {exc.reason} ({exc.source_identifier})")
        return 2

    print_document(document, max(1, args.preview_chars))

    if args.show_text:
        print("\nNormalized text")
        print("---------------")
        print(document.text)

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote complete normalized document to {output}")

    return 0


def main() -> int:
    """CLI entry point."""
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
