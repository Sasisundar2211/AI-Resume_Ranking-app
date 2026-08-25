"""Unit tests for PDF parsing functionality and resource limits."""
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Ensure project_files directory is in Python path for imports if needed
sys.path.append(str(Path(__file__).parent.parent / "project_files" / "Resume_ranking"))

from resume_rankingapp import extract_text_from_pdf


class TestPDFParsing(unittest.TestCase):
    """Test cases for PDF parsing security limits."""

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_from_pdf_page_limit(self, mock_pdf_reader):
        """Test that extract_text_from_pdf stops extracting text after reaching max_pages."""
        # Setup mock pages (e.g., 150 pages)
        mock_pages = []
        for i in range(150):
            page = MagicMock()
            page.extract_text.return_value = f"Page {i} content. "
            mock_pages.append(page)

        mock_instance = MagicMock()
        mock_instance.pages = mock_pages
        mock_pdf_reader.return_value = mock_instance

        # Call with default max_pages=100
        text_default = extract_text_from_pdf("dummy.pdf")

        # Verify page 0 content is included but page 100 content is not
        self.assertIn("Page 0 content.", text_default)
        self.assertIn("Page 99 content.", text_default)
        self.assertNotIn("Page 100 content.", text_default)

        # Verify extract_text was called exactly 100 times
        call_count = sum(page.extract_text.call_count for page in mock_pages)
        self.assertEqual(call_count, 100)

    @patch("resume_rankingapp.PdfReader")
    def test_extract_text_from_pdf_custom_page_limit(self, mock_pdf_reader):
        """Test that extract_text_from_pdf respects custom max_pages."""
        mock_pages = []
        for i in range(10):
            page = MagicMock()
            page.extract_text.return_value = f"Page {i} content. "
            mock_pages.append(page)

        mock_instance = MagicMock()
        mock_instance.pages = mock_pages
        mock_pdf_reader.return_value = mock_instance

        # Call with custom max_pages=5
        text_custom = extract_text_from_pdf("dummy.pdf", max_pages=5)

        self.assertIn("Page 4 content.", text_custom)
        self.assertNotIn("Page 5 content.", text_custom)

        call_count = sum(page.extract_text.call_count for page in mock_pages)
        self.assertEqual(call_count, 5)


if __name__ == "__main__":
    unittest.main()
