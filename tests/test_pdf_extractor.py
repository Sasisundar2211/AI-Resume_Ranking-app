"""Unit tests and benchmark for PDF text extraction."""
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Ensure project_files/Resume_ranking is on the python path
resume_app_dir = Path(__file__).parent.parent / "project_files" / "Resume_ranking"
if str(resume_app_dir) not in sys.path:
    sys.path.insert(0, str(resume_app_dir))

from resume_rankingapp import extract_text_from_pdf


class TestExtractTextFromPdf(unittest.TestCase):
    """Test cases for PDF text extraction."""

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_empty(self, mock_pdf_reader):
        """Test extraction from PDF with no pages."""
        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf_reader.return_value = mock_pdf

        result = extract_text_from_pdf("dummy.pdf")
        self.assertEqual(result, "")

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_single_page(self, mock_pdf_reader):
        """Test extraction from PDF with a single page."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello World"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf

        result = extract_text_from_pdf("dummy.pdf")
        self.assertEqual(result, "Hello World")

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_multiple_pages(self, mock_pdf_reader):
        """Test extraction from PDF with multiple pages."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content. "
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_pdf

        result = extract_text_from_pdf("dummy.pdf")
        self.assertEqual(result, "Page 1 content. Page 2 content.")

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_handles_none(self, mock_pdf_reader):
        """Test extraction when page.extract_text() returns None."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = None

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_pdf

        result = extract_text_from_pdf("dummy.pdf")
        self.assertEqual(result, "Page 1")


if __name__ == "__main__":
    unittest.main()
