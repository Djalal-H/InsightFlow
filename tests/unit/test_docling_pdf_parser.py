"""Tests for the provider-neutral Docling PDF adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.models import DocumentSource
from insightflow.rag.parsers.docling_pdf import (
    DoclingPdfParser,
    PdfProfile,
    _create_docling_converter,
    inspect_pdf,
)


@dataclass
class FakeItem:
    """Minimal Docling-like item used without loading document models."""

    label: str
    text: str = ""
    level: int | None = None
    pages: tuple[int, ...] = ()
    enumerated: bool = False
    marker: str = ""
    language: str | None = None
    table_text: str = ""
    rows: int | None = None
    columns: int | None = None

    @property
    def prov(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(page_no=page) for page in self.pages]

    @property
    def data(self) -> SimpleNamespace:
        return SimpleNamespace(num_rows=self.rows, num_cols=self.columns)

    def export_to_markdown(self, *, doc: object | None = None) -> str:
        assert doc is not None
        return self.table_text


class FakeDocument:
    """Docling-like document that returns items in deterministic reading order."""

    def __init__(self, items: list[tuple[FakeItem, int]]) -> None:
        self.items = items

    def iterate_items(self) -> list[tuple[FakeItem, int]]:
        return self.items


class FakeConverter:
    """Record conversion calls and return a configured result."""

    def __init__(self, document: FakeDocument | None, *, status: str = "success") -> None:
        self.result = SimpleNamespace(
            status=SimpleNamespace(value=status),
            document=document,
        )
        self.calls: list[tuple[Path, bool]] = []

    def convert(self, source: Path, *, raises_on_error: bool) -> object:
        self.calls.append((source, raises_on_error))
        return self.result


def make_source(tmp_path: Path, *, mime_type: str = "application/pdf") -> DocumentSource:
    """Create a source whose contents are bypassed by the injected inspector."""
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\n% fake unit-test input\n")
    return DocumentSource(
        source_name=path.name,
        source_identifier="papers/paper.pdf",
        path=path,
        mime_type=mime_type,
        metadata={"corpus": "evaluation"},
    )


def digital_profile(path: Path) -> PdfProfile:
    """Return a reusable digital-PDF profile for adapter tests."""
    del path
    return PdfProfile(page_count=3, encrypted=False, has_images=False)


async def inline_conversion(converter: object, source: Path) -> object:
    """Run fake conversion inline so unit tests never create worker threads."""
    assert isinstance(converter, FakeConverter)
    return converter.convert(source, raises_on_error=False)


def structural_items() -> list[tuple[FakeItem, int]]:
    """Build representative Docling output spanning structure and pages."""
    return [
        (FakeItem("title", "Retrieval Evaluation", pages=(1,)), 1),
        (FakeItem("section_header", "Introduction", level=1, pages=(1,)), 1),
        (FakeItem("text", "The study compares retrievers.", pages=(1,)), 2),
        (FakeItem("section_header", "Method", level=2, pages=(2,)), 2),
        (
            FakeItem(
                "list_item",
                "Index every document.",
                pages=(2,),
                enumerated=True,
                marker="1.",
            ),
            2,
        ),
        (
            FakeItem(
                "table",
                table_text="| Method | Recall |\n| --- | --- |\n| Dense | 0.81 |",
                pages=(2, 3),
                rows=2,
                columns=2,
            ),
            2,
        ),
        (FakeItem("caption", "Table 1: Retrieval results", pages=(3,)), 2),
        (FakeItem("code", "query = embed(text)", pages=(3,), language="python"), 2),
        (FakeItem("page_footer", "Conference 2026", pages=(3,)), 1),
        (FakeItem("picture", pages=(3,)), 1),
    ]


@pytest.mark.asyncio
async def test_parser_maps_structure_hierarchy_tables_and_provenance(tmp_path: Path) -> None:
    """Docling values become ordered provider-neutral domain records."""
    source = make_source(tmp_path)
    converter = FakeConverter(FakeDocument(structural_items()))
    parser = DoclingPdfParser(
        converter=converter,
        pdf_inspector=digital_profile,
        conversion_runner=inline_conversion,
    )

    document = await parser.parse(source)

    assert converter.calls == [(source.path, False)]
    assert document.title == "Retrieval Evaluation"
    assert document.parser_name == "docling_pdf"
    assert document.metadata == {"corpus": "evaluation", "page_count": 3}
    assert [element.element_type for element in document.elements] == [
        "title",
        "heading",
        "paragraph",
        "heading",
        "list_item",
        "table",
        "caption",
        "code",
        "other",
    ]

    introduction, paragraph, method = document.elements[1:4]
    assert introduction.heading_path == ["Introduction"]
    assert introduction.metadata == {"heading_level": 1}
    assert paragraph.heading_path == ["Introduction"]
    assert method.heading_path == ["Introduction", "Method"]

    list_item = document.elements[4]
    assert list_item.heading_path == ["Introduction", "Method"]
    assert list_item.metadata == {
        "list_kind": "ordered",
        "list_depth": 2,
        "ordinal": 1,
    }

    table = document.elements[5]
    assert table.page_number == 2
    assert table.metadata == {
        "row_count": 2,
        "column_count": 2,
        "page_numbers": [2, 3],
    }
    assert table.text.startswith("| Method | Recall |")
    assert document.elements[7].metadata == {"language": "python"}
    assert document.elements[8].metadata == {"source_role": "page_footer"}
    assert all(
        document.text[element.char_start : element.char_end] == element.text
        for element in document.elements
        if element.char_start is not None and element.char_end is not None
    )
    assert "FakeItem" not in document.model_dump_json()


@pytest.mark.asyncio
async def test_parser_output_is_deterministic(tmp_path: Path) -> None:
    """The same source and mapped content produce identical identities."""
    source = make_source(tmp_path)
    parser = DoclingPdfParser(
        converter=FakeConverter(FakeDocument(structural_items())),
        pdf_inspector=digital_profile,
        conversion_runner=inline_conversion,
    )

    first = await parser.parse(source)
    second = await parser.parse(source)

    assert first.document_id == second.document_id
    assert first.document_version == second.document_version
    assert [item.element_id for item in first.elements] == [
        item.element_id for item in second.elements
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "expected_reason"),
    [
        (PdfProfile(page_count=1, encrypted=True, has_images=False), "encrypted_pdf"),
        (PdfProfile(page_count=1, encrypted=False, has_images=True), "scanned_pdf"),
        (PdfProfile(page_count=1, encrypted=False, has_images=False), "textless_pdf"),
    ],
)
async def test_parser_rejects_unusable_pdfs(
    tmp_path: Path,
    profile: PdfProfile,
    expected_reason: str,
) -> None:
    """Encrypted, scanned, and textless PDFs have stable rejection reasons."""
    source = make_source(tmp_path)
    parser = DoclingPdfParser(
        converter=FakeConverter(FakeDocument([])),
        pdf_inspector=lambda path: profile,
        conversion_runner=inline_conversion,
    )

    with pytest.raises(DocumentRejectedError) as error:
        await parser.parse(source)

    assert error.value.reason == expected_reason


@pytest.mark.asyncio
async def test_parser_rejects_wrong_format_and_failed_conversion(tmp_path: Path) -> None:
    """The adapter does not leak unsupported formats or Docling failure states."""
    wrong_source = make_source(tmp_path, mime_type="text/plain")
    parser = DoclingPdfParser(
        converter=FakeConverter(FakeDocument(structural_items())),
        pdf_inspector=digital_profile,
        conversion_runner=inline_conversion,
    )

    with pytest.raises(DocumentRejectedError) as wrong_error:
        await parser.parse(wrong_source)
    assert wrong_error.value.reason == "unsupported_format"

    source = make_source(tmp_path)
    failed_parser = DoclingPdfParser(
        converter=FakeConverter(None, status="failure"),
        pdf_inspector=digital_profile,
        conversion_runner=inline_conversion,
    )
    with pytest.raises(DocumentRejectedError) as failed_error:
        await failed_parser.parse(source)
    assert failed_error.value.reason == "conversion_failed"


def test_real_converter_explicitly_disables_ocr_and_enables_tables() -> None:
    """The default Docling configuration enforces the digital-PDF policy."""
    from docling.datamodel.base_models import InputFormat

    converter = _create_docling_converter()
    options = converter.format_to_options[InputFormat.PDF].pipeline_options  # type: ignore[attr-defined]

    assert options.do_ocr is False
    assert options.do_table_structure is True


def test_pdf_inspector_detects_encryption(tmp_path: Path) -> None:
    """Encrypted documents are identified before Docling conversion."""
    plain_path = tmp_path / "plain.pdf"
    encrypted_path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with plain_path.open("wb") as stream:
        writer.write(stream)

    encrypted_writer = PdfWriter(clone_from=plain_path)
    encrypted_writer.encrypt("secret")
    with encrypted_path.open("wb") as stream:
        encrypted_writer.write(stream)

    assert inspect_pdf(plain_path) == PdfProfile(
        page_count=1,
        encrypted=False,
        has_images=False,
    )
    assert inspect_pdf(encrypted_path).encrypted is True
