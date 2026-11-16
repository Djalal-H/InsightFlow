"""The API process must not load the ingestion-only document stack."""

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
INGEST_ONLY = ("docling", "pypdf", "markdown", "torch")


def test_api_import_does_not_load_ingestion_packages() -> None:
    # A new interpreter: the test process itself has these modules from other tests.
    code = (
        "import json, sys; import insightflow.main; "
        f"print(json.dumps([name for name in {list(INGEST_ONLY)!r} if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        cwd=SRC,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
        text=True,
        timeout=120,
    )

    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_lazy_rag_exports_still_resolve() -> None:
    import insightflow.rag as rag
    from insightflow.rag.chunkers.max_min import MaxMinSemanticChunker
    from insightflow.rag.parsers.docling_pdf import DoclingPdfParser

    assert rag.DoclingPdfParser is DoclingPdfParser
    assert rag.MaxMinSemanticChunker is MaxMinSemanticChunker
