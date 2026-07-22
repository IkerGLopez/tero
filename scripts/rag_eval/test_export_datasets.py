"""
Tests for export_datasets.py — manual-only utility guard.
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestManualFlag:
    """Manual-acknowledgement guard behavior."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import export_datasets as ed
        self._ed = ed

    def test_default_run_exits_without_manual(self, tmp_path: Path):
        """Without --manual the script exits asking for acknowledgement."""
        with patch("sys.argv", ["export_datasets.py"]):
            with pytest.raises(SystemExit) as exc_info:
                self._ed.main()
        assert exc_info.value.code == 0

    def test_manual_flag_allows_export(self, tmp_path: Path):
        """With --manual the export runs."""
        with patch.object(self._ed, "export_dataset") as mock_export:
            with patch("sys.argv", ["export_datasets.py", "--manual", "--dataset", "ragbench"]):
                self._ed.main()

        mock_export.assert_called_once_with("ragbench", n=10, corpus_size=200)


def test_export_dataset_signature_unchanged():
    """export_dataset is still callable from smoke_test."""
    import export_datasets as ed
    assert callable(ed.export_dataset)
