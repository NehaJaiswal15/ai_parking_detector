"""
Tests for the OccupancyLogger class.
Validates CSV file creation, header writing, and data logging.
"""

import csv
import os
import tempfile

import pytest

from src.occupancy_logger import OccupancyLogger


@pytest.fixture
def temp_csv(tmp_path):
    """Provide a temporary CSV file path for testing."""
    return str(tmp_path / "test_occupancy.csv")


class TestOccupancyLogger:
    """Tests for the OccupancyLogger class."""

    def test_creates_csv_with_headers(self, temp_csv):
        """A new logger should create a CSV file with the correct headers."""
        logger = OccupancyLogger(csv_path=temp_csv)

        assert os.path.exists(temp_csv)
        with open(temp_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert headers == OccupancyLogger.HEADERS

    def test_logs_single_row(self, temp_csv):
        """Logging one entry should add exactly one data row after the header."""
        logger = OccupancyLogger(csv_path=temp_csv)
        logger.log(frame_num=1, total_slots=10, occupied=3, available=7, occupancy_pct=30.0)

        with open(temp_csv, "r") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 2  # header + 1 data row
        assert rows[1][1] == "1"       # frame_num
        assert rows[1][2] == "10"      # total_slots
        assert rows[1][3] == "3"       # occupied
        assert rows[1][4] == "7"       # available
        assert rows[1][5] == "30.0"    # occupancy_pct

    def test_logs_multiple_rows(self, temp_csv):
        """Multiple log calls should append rows sequentially."""
        logger = OccupancyLogger(csv_path=temp_csv)
        for i in range(5):
            logger.log(frame_num=i, total_slots=10, occupied=i, available=10 - i, occupancy_pct=i * 10.0)

        with open(temp_csv, "r") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 6  # header + 5 data rows

    def test_timestamp_is_present(self, temp_csv):
        """Each logged row should have a non-empty timestamp."""
        logger = OccupancyLogger(csv_path=temp_csv)
        logger.log(frame_num=1, total_slots=5, occupied=2, available=3, occupancy_pct=40.0)

        with open(temp_csv, "r") as f:
            rows = list(csv.reader(f))

        timestamp = rows[1][0]
        assert len(timestamp) > 0
        assert "-" in timestamp  # basic check for date format
