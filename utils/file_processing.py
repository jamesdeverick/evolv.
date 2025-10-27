# --------------------------------------------
# File Processing Utilities
# --------------------------------------------

import io
import zipfile
from defusedxml import ElementTree as ET
from config import MAX_UPLOAD_SIZE

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def validate_file_size(file_bytes: bytes) -> bool:
    """Validate that file size is within limits."""
    return len(file_bytes) <= MAX_UPLOAD_SIZE


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Lightweight DOCX extractor using defusedxml for security.

    Args:
        file_bytes: Raw bytes of the DOCX file

    Returns:
        Extracted text content
    """
    if not validate_file_size(file_bytes):
        return "[File too large. Maximum 10MB allowed.]"

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            # Validate that it's actually a DOCX file
            if "word/document.xml" not in z.namelist():
                return "[Invalid DOCX file structure.]"

            with z.open("word/document.xml") as f:
                xml = f.read()

        # Use defusedxml to prevent XML bombs
        root = ET.fromstring(xml)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        texts = []
        for para in root.findall(".//w:p", ns):
            runs = []
            for t in para.findall(".//w:t", ns):
                if t.text:
                    runs.append(t.text)
            line = "".join(runs).strip()
            if line:
                texts.append(line)

        return "\n".join(texts)
    except zipfile.BadZipFile:
        return "[Invalid ZIP/DOCX file.]"
    except ET.ParseError:
        return "[XML parsing error. Corrupted DOCX file.]"
    except Exception as e:
        return f"[DOCX read error: {e}]"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF files.

    Args:
        file_bytes: Raw bytes of the PDF file

    Returns:
        Extracted text content
    """
    if not validate_file_size(file_bytes):
        return "[File too large. Maximum 10MB allowed.]"

    if PyPDF2 is None:
        return "[PDF parsing not available. Install PyPDF2 or upload TXT/MD/DOCX.]"

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        return f"[PDF read error: {e}]"


def read_tov_upload(uploaded) -> str:
    """
    Read Tone of Voice file into plain text.

    Args:
        uploaded: Streamlit UploadedFile object

    Returns:
        Extracted text content
    """
    if uploaded is None:
        return ""

    name = (uploaded.name or "").lower()
    data = uploaded.read()

    # Route to appropriate extractor
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    elif name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    else:
        # Assume text file (txt, md, or unknown)
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return "[Could not decode file as UTF-8 text.]"


def truncate_text(text: str, max_length: int = 50000) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum characters to keep

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) > max_length:
        return text[:max_length] + "\n\n... [Content truncated due to length] ..."
    return text
