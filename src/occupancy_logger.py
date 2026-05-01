"""
Occupancy Logger — writes timestamped parking occupancy data to CSV.
Used by both the CLI (main.py) and the Streamlit dashboard for
historical trend analysis.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from src.config import CSV_LOG_FILE, OUTPUT_DIR

logger = logging.getLogger(__name__)


class OccupancyLogger:
    """Append-mode CSV logger for parking occupancy data.

    Creates the output directory and CSV file automatically.
    Each row logs: timestamp, frame number, total slots, occupied,
    available, and occupancy percentage.

    Args:
        csv_path: Path to the output CSV file.
    """

    HEADERS = ["timestamp", "frame", "total_slots", "occupied", "available", "occupancy_pct"]

    def __init__(self, csv_path: str = CSV_LOG_FILE) -> None:
        self.csv_path = csv_path
        self._ensure_dir()
        self._write_header()
        logger.info("Occupancy logger initialized: %s", csv_path)

    def _ensure_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def _write_header(self) -> None:
        """Write CSV header if the file is new or empty."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log(
        self,
        frame_num: int,
        total_slots: int,
        occupied: int,
        available: int,
        occupancy_pct: float,
    ) -> None:
        """Append a single occupancy record to the CSV.

        Args:
            frame_num: Current video frame number.
            total_slots: Total parking slots defined.
            occupied: Number of occupied slots.
            available: Number of available slots.
            occupancy_pct: Occupancy as a percentage (0–100).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_num, total_slots, occupied, available, f"{occupancy_pct:.1f}"])
