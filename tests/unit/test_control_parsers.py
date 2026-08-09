"""Tests for the Track A (control) TXT, Markdown, and DOCX parsers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from insightflow.core.exceptions import DocumentRejectedError
from insightflow.rag.models import DocumentSource
from insightflow.rag.parsers.docx import DocxParser
from insightflow.rag.parsers.markdown import MarkdownParser
from insightflow.rag.parsers.txt import TxtParser

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---- TxtParser ----------------------------------------------------------

def make_txt_source(
    tmp_path: Path, content: str, *, mime_type: str = "text/plain"
) -> DocumentSource:
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")
    return DocumentSource(
        source_name=path.name,
        source_identifier="notes/sample.txt",
        path=path,
        mime_type=mime_type,
    )


@pytest.mark.asyncio
async def test_txt_parser_normalizes_whitespace(tmp_path: Path) -> None:
    """Trailing spaces and excess blank lines are collapsed, content preserved."""
    source = make_txt_source(tmp_path, "Hello world.   \n\n\n\nSecond line.\n")
    document = await TxtParser().parse(source)

    assert document.text == "Hello world.\n\nSecond line."
    assert len(document.elements) == 1
    assert document.elements[0].order == 0


@pytest.mark.asyncio
async def test_txt_parser_is_deterministic(tmp_path: Path) -> None:
    """Parsing the same source twice produces identical identities."""
    source = make_txt_source(tmp_path, "Stable content.")
    first = await TxtParser().parse(source)
    second = await TxtParser().parse(source)

    assert first.document_id == second.document_id
    assert first.document_version == second.document_version
    assert first.elements[0].element_id == second.elements[0].element_id


@pytest.mark.asyncio
async def test_txt_parser_rejects_wrong_mime_type(tmp_path: Path) -> None:
    """A source claiming a non-text/plain mime type is rejected before reading."""
    source = make_txt_source(tmp_path, "content", mime_type="application/pdf")

    with pytest.raises(DocumentRejectedError) as error:
        await TxtParser().parse(source)

    assert error.value.reason == "unsupported_format"


@pytest.mark.asyncio
async def test_txt_parser_rejects_empty_file(tmp_path: Path) -> None:
    """A file with only whitespace has no usable content."""
    source = make_txt_source(tmp_path, "   \n\n  \n")

    with pytest.raises(DocumentRejectedError) as error:
        await TxtParser().parse(source)

    assert error.value.reason == "empty_document"


# ---- MarkdownParser -------------------------------------------------------


def make_md_source(
    tmp_path: Path, content: str, *, mime_type: str = "text/markdown"
) -> DocumentSource:
    path = tmp_path / "sample.md"
    path.write_text(content, encoding="utf-8")
    return DocumentSource(
        source_name=path.name,
        source_identifier="docs/sample.md",
        path=path,
        mime_type=mime_type,
    )


@pytest.mark.asyncio
async def test_markdown_parser_flattens_and_extracts_title(tmp_path: Path) -> None:
    """Markdown structure is stripped to plain text; the H1 heading becomes the title."""
    source = make_md_source(
        tmp_path,
        "# My Title\n\nSome **bold** text and a list:\n\n- one\n- two\n",
    )
    document = await MarkdownParser().parse(source)

    assert document.title == "My Title"
    assert "**" not in document.text
    assert "My Title" in document.text
    assert "one" in document.text


@pytest.mark.asyncio
async def test_markdown_parser_strips_heading_anchor_syntax(tmp_path: Path) -> None:
    """Doc-generator heading anchors like `{ #id }` do not leak into the title."""
    source = make_md_source(tmp_path, "# Request Body { #request-body }\n\nBody text.\n")
    document = await MarkdownParser().parse(source)

    assert document.title == "Request Body"


@pytest.mark.asyncio
async def test_markdown_parser_title_is_none_without_h1(tmp_path: Path) -> None:
    """Documents with no top-level Markdown heading have no title, not a crash."""
    source = make_md_source(tmp_path, "## Only a subheading\n\nSome text.\n")
    document = await MarkdownParser().parse(source)

    assert document.title is None


@pytest.mark.asyncio
async def test_markdown_parser_rejects_wrong_mime_type(tmp_path: Path) -> None:
    source = make_md_source(tmp_path, "# Title\n\nText.", mime_type="text/plain")

    with pytest.raises(DocumentRejectedError) as error:
        await MarkdownParser().parse(source)

    assert error.value.reason == "unsupported_format"


@pytest.mark.asyncio
async def test_markdown_parser_rejects_empty_document(tmp_path: Path) -> None:
    """Markdown that flattens to nothing (e.g. only comments) is rejected."""
    source = make_md_source(tmp_path, "<!-- just a comment -->")

    with pytest.raises(DocumentRejectedError) as error:
        await MarkdownParser().parse(source)

    assert error.value.reason == "empty_document"


# ---- DocxParser -----------------------------------------------------------


@dataclass
class FakeDocxItem:
    """Minimal Docling-like text item used without loading real DOCX models."""

    text: str = ""


class FakeDocxDocument:
    """Docling-like document exposing texts in deterministic reading order."""

    def __init__(self, texts: list[FakeDocxItem]) -> None:
        self.texts = texts


class FakeDocxConverter:
    """Record conversion calls and return a configured result."""

    def __init__(self, document: FakeDocxDocument | None, *, status: str = "success") -> None:
        self.result = SimpleNamespace(status=SimpleNamespace(value=status), document=document)
        self.calls: list[tuple[Path, bool]] = []

    def convert(self, source: Path, *, raises_on_error: bool) -> object:
        self.calls.append((source, raises_on_error))
        return self.result


async def inline_conversion(converter: object, source: Path) -> object:
    """Run fake conversion inline so unit tests never create worker threads."""
    assert isinstance(converter, FakeDocxConverter)
    return converter.convert(source, raises_on_error=False)


def make_docx_source(tmp_path: Path, *, mime_type: str = DOCX_MIME) -> DocumentSource:
    path = tmp_path / "report.docx"
    path.write_bytes(b"PK\x03\x04 fake docx bytes")
    return DocumentSource(
        source_name=path.name,
        source_identifier="reports/report.docx",
        path=path,
        mime_type=mime_type,
    )


@pytest.mark.asyncio
async def test_docx_parser_flattens_texts_in_order(tmp_path: Path) -> None:
    """Docling text items become ordered, traceable elements."""
    source = make_docx_source(tmp_path)
    document_body = FakeDocxDocument(
        [FakeDocxItem("Title"), FakeDocxItem("First paragraph."), FakeDocxItem("Second paragraph.")]
    )
    converter = FakeDocxConverter(document_body)
    parser = DocxParser(converter=converter, conversion_runner=inline_conversion)

    document = await parser.parse(source)

    assert converter.calls == [(source.path, False)]
    assert [element.text for element in document.elements] == [
        "Title",
        "First paragraph.",
        "Second paragraph.",
    ]
    assert [element.order for element in document.elements] == [0, 1, 2]
    assert all(element.page_number is None for element in document.elements)


@pytest.mark.asyncio
async def test_docx_parser_skips_blank_text_items(tmp_path: Path) -> None:
    """Whitespace-only Docling items are dropped rather than becoming empty elements."""
    source = make_docx_source(tmp_path)
    document_body = FakeDocxDocument(
        [FakeDocxItem("Real content."), FakeDocxItem("   "), FakeDocxItem("")]
    )
    parser = DocxParser(
        converter=FakeDocxConverter(document_body),
        conversion_runner=inline_conversion,
    )

    document = await parser.parse(source)

    assert len(document.elements) == 1
    assert document.elements[0].text == "Real content."


@pytest.mark.asyncio
async def test_docx_parser_is_deterministic(tmp_path: Path) -> None:
    """The same source and mapped content produce identical identities."""
    source = make_docx_source(tmp_path)
    document_body = FakeDocxDocument([FakeDocxItem("Stable content.")])
    parser = DocxParser(
        converter=FakeDocxConverter(document_body),
        conversion_runner=inline_conversion,
    )

    first = await parser.parse(source)
    second = await parser.parse(source)

    assert first.document_id == second.document_id
    assert first.document_version == second.document_version
    assert first.elements[0].element_id == second.elements[0].element_id


@pytest.mark.asyncio
async def test_docx_parser_rejects_wrong_mime_type(tmp_path: Path) -> None:
    source = make_docx_source(tmp_path, mime_type="application/pdf")
    parser = DocxParser(
        converter=FakeDocxConverter(FakeDocxDocument([FakeDocxItem("text")])),
        conversion_runner=inline_conversion,
    )

    with pytest.raises(DocumentRejectedError) as error:
        await parser.parse(source)

    assert error.value.reason == "unsupported_format"


@pytest.mark.asyncio
async def test_docx_parser_rejects_empty_document(tmp_path: Path) -> None:
    """A DOCX with no extractable text content is rejected, not silently empty."""
    source = make_docx_source(tmp_path)
    parser = DocxParser(
        converter=FakeDocxConverter(FakeDocxDocument([])),
        conversion_runner=inline_conversion,
    )

    with pytest.raises(DocumentRejectedError) as error:
        await parser.parse(source)

    assert error.value.reason == "empty_document"


@pytest.mark.asyncio
async def test_docx_parser_rejects_failed_conversion(tmp_path: Path) -> None:
    """A Docling conversion failure is surfaced as a stable rejection reason."""
    source = make_docx_source(tmp_path)
    parser = DocxParser(
        converter=FakeDocxConverter(None, status="failure"),
        conversion_runner=inline_conversion,
    )

    with pytest.raises(DocumentRejectedError) as error:
        await parser.parse(source)

    assert error.value.reason == "conversion_failed"
