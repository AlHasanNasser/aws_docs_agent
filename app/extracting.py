from pathlib import Path
from collections.abc import Iterator
from typing import Any

import pdfplumber


def _cell_text(value: Any) -> str:
    """Normalize a PDF table cell into stable, single-line text."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _table_to_markdown(table: list[list[Any]]) -> str:
    """Convert a pdfplumber table into markdown without requiring tabulate."""
    rows = [[_cell_text(cell) for cell in row] for row in table if row]
    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    header = rows[0]
    body = rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_pdf_content(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract text and tables from one PDF as chunk-ready records.

    Text inside detected table bounds is excluded so table content is not
    duplicated in the text record. Each record retains source and page data
    needed for retrieval citations after chunking.
    """
    return list(iter_pdf_content(pdf_path))


def iter_pdf_content(pdf_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield extracted records page by page without retaining the whole PDF."""
    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            def not_in_tables(obj: dict[str, Any]) -> bool:
                return not any(
                    obj["x0"] >= bbox[0]
                    and obj["top"] >= bbox[1]
                    and obj["x1"] <= bbox[2]
                    and obj["bottom"] <= bbox[3]
                    for bbox in table_bboxes
                )

            clean_text = page.filter(not_in_tables).extract_text()
            if clean_text and clean_text.strip():
                yield {
                    "page": page_num,
                    "type": "text",
                    "content": clean_text.strip(),
                    "source": str(pdf_path),
                    "metadata": {"source": str(pdf_path), "page": page_num, "type": "text"},
                }

            for table_index, table in enumerate(tables, start=1):
                content = _table_to_markdown(table.extract())
                if content:
                    yield {
                        "page": page_num,
                        "type": "table",
                        "content": content,
                        "source": str(pdf_path),
                        "metadata": {
                            "source": str(pdf_path),
                            "page": page_num,
                            "type": "table",
                            "table_index": table_index,
                            "bbox": table.bbox,
                        },
                    }


def extract_pdfs_from_folder(
    folder_path: str | Path = "docs",
    *,
    recursive: bool = True,
) -> list[dict[str, Any]]:
    """Extract every PDF under a folder in deterministic path order."""
    folder_path = Path(folder_path)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(
        path for path in folder_path.glob(pattern) if path.is_file()
    )

    return list(iter_pdfs_from_folder(folder_path, recursive=recursive))


def iter_pdfs_from_folder(
    folder_path: str | Path = "docs",
    *,
    recursive: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield records from every PDF in deterministic path order."""
    folder_path = Path(folder_path)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(path for path in folder_path.glob(pattern) if path.is_file())

    for pdf_path in pdf_paths:
        yield from iter_pdf_content(pdf_path)