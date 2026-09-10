import pytest

from health_agents.io_utils import UnsupportedSource, resolve_source


@pytest.fixture
def report_file(tmp_path):
    path = tmp_path / "bloodwork.txt"
    path.write_text("Haemoglobin 9.1 g/dL (low)\n", encoding="utf-8")
    return path


def test_reads_a_txt_path_embedded_in_a_sentence(report_file):
    assert resolve_source(f"summarize {report_file} please") == "Haemoglobin 9.1 g/dL (low)\n"


def test_reads_a_bare_txt_path(report_file):
    assert resolve_source(str(report_file)) == "Haemoglobin 9.1 g/dL (low)\n"


def test_strips_surrounding_quotes(report_file):
    assert resolve_source(f'summarize "{report_file}"') is not None


def test_returns_none_when_no_path_present():
    assert resolve_source("summarize my health report") is None


def test_returns_none_for_prose_that_merely_mentions_txt():
    assert resolve_source("I have notes.txt somewhere but not here") is None


def test_returns_none_for_empty_input():
    assert resolve_source("") is None
    assert resolve_source(None) is None


def test_rejects_a_non_txt_file_that_exists(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(UnsupportedSource, match="\\.txt"):
        resolve_source(f"summarize {pdf}")


def test_rejects_an_empty_txt_file(tmp_path):
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n", encoding="utf-8")
    with pytest.raises(UnsupportedSource, match="empty"):
        resolve_source(f"summarize {blank}")


def test_directory_is_not_mistaken_for_a_file(tmp_path):
    directory = tmp_path / "reports.txt"
    directory.mkdir()
    assert resolve_source(f"summarize {directory}") is None
