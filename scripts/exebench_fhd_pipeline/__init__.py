"""ExeBench-FHD per-record pipeline: adds every main-build field to each record
of a split in one pass, instead of running the scripts/extract/extract_*.py
scripts separately and merging their outputs.

Run: python3 -m scripts.exebench_fhd_pipeline --split ...
"""
from scripts.exebench_fhd_pipeline.core import process_record

__all__ = ["process_record"]
