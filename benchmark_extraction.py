"""Benchmark for PDF text extraction performance."""
import sys
import time
from pathlib import Path

resume_app_dir = Path(__file__).parent / "project_files" / "Resume_ranking"
if str(resume_app_dir) not in sys.path:
    sys.path.insert(0, str(resume_app_dir))


class MockPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class MockPdf:
    def __init__(self, num_pages, text_per_page):
        self.pages = [MockPage(text_per_page) for _ in range(num_pages)]


def original_extract(pdf):
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text


def optimized_extract(pdf):
    return "".join(page.extract_text() or "" for page in pdf.pages)


def run_benchmark(num_pages=20000, text_len=1000, iterations=5):
    page_text = "Standard resume line text content " * 10
    mock_pdf = MockPdf(num_pages, page_text)

    # Benchmark Original
    orig_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        original_extract(mock_pdf)
        t1 = time.perf_counter()
        orig_times.append(t1 - t0)

    # Benchmark Optimized
    opt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        optimized_extract(mock_pdf)
        t1 = time.perf_counter()
        opt_times.append(t1 - t0)

    avg_orig = sum(orig_times) / len(orig_times)
    avg_opt = sum(opt_times) / len(opt_times)
    speedup = avg_orig / avg_opt if avg_opt > 0 else 0

    print(f"--- Benchmark Results ({num_pages:,} pages) ---")
    print(f"Original  (+= loop):  {avg_orig*1000:.2f} ms")
    print(f"Optimized (''.join):  {avg_opt*1000:.2f} ms")
    print(f"Speedup:              {speedup:.2f}x faster")
    print(f"Time saved:           {(avg_orig - avg_opt)*1000:.2f} ms")


if __name__ == "__main__":
    run_benchmark(num_pages=30000)
