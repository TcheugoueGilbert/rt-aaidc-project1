import os
from pathlib import Path
from typing import List, Dict, Any


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import PyPDF2

        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception:
        try:
            # fallback to pdftotext if available
            import subprocess

            out = subprocess.check_output(["pdftotext", str(path), "-"])  # may fail
            return out.decode("utf-8", errors="ignore")
        except Exception:
            print(f"[loader] Could not parse PDF {path} (install PyPDF2 or pdftotext)")
            return ""


def _read_docx(path: Path) -> str:
    try:
        import docx

        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        print(f"[loader] Could not parse DOCX {path} (install python-docx)")
        return ""


def _read_csv(path: Path) -> str:
    try:
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_csv(index=False)
    except Exception:
        # fallback: simple csv join
        try:
            import csv

            rows = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for r in reader:
                    rows.append(" ".join(r))
            return "\n".join(rows)
        except Exception:
            print(f"[loader] Could not parse CSV {path} (install pandas for richer parsing)")
            return ""


def _read_excel(path: Path) -> str:
    try:
        import pandas as pd

        df = pd.read_excel(path)
        return df.to_csv(index=False)
    except Exception:
        print(f"[loader] Could not parse Excel {path} (install pandas and openpyxl)")
        return ""


def load_documents_from_dir(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load documents from a directory. Supports .txt, .pdf, .docx, .csv, .xls, .xlsx

    Returns a list of dicts: {"content": str, "metadata": {"source": filename}}
    """
    data_path = Path(data_dir)
    docs: List[Dict[str, Any]] = []
    if not data_path.exists():
        return docs

    for p in sorted(data_path.iterdir()):
        if p.is_dir():
            continue
        suffix = p.suffix.lower()
        content = ""
        if suffix == ".txt":
            content = _read_txt(p)
        elif suffix == ".pdf":
            content = _read_pdf(p)
        elif suffix in (".docx",):
            content = _read_docx(p)
        elif suffix == ".csv":
            content = _read_csv(p)
        elif suffix in (".xls", ".xlsx"):
            content = _read_excel(p)
        else:
            # ignore unknown types
            continue

        if content and content.strip():
            docs.append({"content": content, "metadata": {"source": p.name}})

    return docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load documents from a directory and print a summary")
    parser.add_argument("--dir", "-d", default=str(Path(__file__).resolve().parents[1] / "data"), help="Directory to load documents from")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print document snippets")
    args = parser.parse_args()

    loaded = load_documents_from_dir(args.dir)
    print(f"[loader] Loaded {len(loaded)} documents from {args.dir}")
    if args.verbose and loaded:
        for i, doc in enumerate(loaded, 1):
            src = doc.get("metadata", {}).get("source", "<unknown>")
            snippet = doc.get("content", "")[:400].replace("\n", " ")
            print(f"\n[{i}] {src} — {len(doc.get('content',''))} chars")
            print(snippet)
