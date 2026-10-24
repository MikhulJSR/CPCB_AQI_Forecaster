"""
Unit tests for CPCB Ingestion Client and Schema Validation.
"""

import pytest
import pandas as pd
from src.ingestion.schema import CPCBRawRecord
from src.ingestion.cpcb_client import CPCBClient


def test_cpcb_raw_record_schema_valid():
    """Verify that a valid CPCB record passes Pydantic validation."""
    payload = {
        "country": "India",
        "state": "Delhi",
        "city": "Delhi",
        "station": "Anand Vihar, Delhi - DPCC",
        "last_update": "24-10-2026 18:00:00",
        "pollutant_id": "PM2.5",
        "pollutant_min": "180.0",
        "pollutant_max": "320.0",
        "pollutant_avg": "245.5",
        "pollutant_unit": "ug/m3"
    }
    record = CPCBRawRecord(**payload)
    assert record.pollutant_avg == 245.5
    assert record.station == "Anand Vihar, Delhi - DPCC"


def test_cpcb_raw_record_schema_na_handling():
    """Verify that NA / missing numeric values are parsed into None without crashing."""
    payload = {
        "city": "Delhi",
        "station": "ITO, Delhi - CPCB",
        "last_update": "24-10-2026 19:00:00",
        "pollutant_id": "SO2",
        "pollutant_avg": "NA"
    }
    record = CPCBRawRecord(**payload)
    assert record.pollutant_avg is None


def test_parse_and_pivot_long_to_wide():
    """Verify that long-format telemetry records pivot properly into feature columns."""
    client = CPCBClient()
    sample_records = [
        {
            "country": "India", "state": "Delhi", "city": "Delhi",
            "station": "Anand Vihar", "last_update": "24-10-2026 10:00:00",
            "pollutant_id": "PM2.5", "pollutant_avg": "150.0"
        },
        {
            "country": "India", "state": "Delhi", "city": "Delhi",
            "station": "Anand Vihar", "last_update": "24-10-2026 10:00:00",
            "pollutant_id": "NO2", "pollutant_avg": "45.0"
        }
    ]
    df_wide = client.parse_and_pivot(sample_records)
    assert len(df_wide) == 1
    assert "pm25" in df_wide.columns
    assert "no2" in df_wide.columns
    assert df_wide.iloc[0]["pm25"] == 150.0
    assert df_wide.iloc[0]["no2"] == 45.0
