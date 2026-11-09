import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader

class PDFExtractor:
    @staticmethod
    def extract_pages_from_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from PDF page by page, preserving 1-indexed page numbers.
        Returns:
            List of dicts: [{"page_number": int, "text": str}]
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_data = []
        
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            # Clean text: remove null bytes, fix excessive whitespace while preserving paragraph breaks
            cleaned_text = re.sub(r'\x00', '', raw_text)
            cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
            
            if cleaned_text:
                pages_data.append({
                    "page_number": page_num,
                    "text": cleaned_text
                })
        return pages_data

    @staticmethod
    def extract_pages_from_file(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "rb") as f:
            return PDFExtractor.extract_pages_from_bytes(f.read())
