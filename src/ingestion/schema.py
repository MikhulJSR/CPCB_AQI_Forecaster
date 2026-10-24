"""
Data contract and Pydantic validation schema for CPCB / data.gov.in telemetry.
Ensures strong typing and early detection of API payload schema changes.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CPCBRawRecord(BaseModel):
    """Schema representing an individual raw telemetry row from the data.gov.in CPCB API."""
    country: Optional[str] = "India"
    state: Optional[str] = None
    city: str
    station: str
    last_update: str = Field(..., description="Timestamp string from the monitor, e.g. 24-10-2026 18:00:00")
    pollutant_id: str = Field(..., description="E.g. PM2.5, PM10, NO2, SO2, CO, OZONE, NH3")
    pollutant_min: Optional[float] = None
    pollutant_max: Optional[float] = None
    pollutant_avg: Optional[float] = None
    pollutant_unit: Optional[str] = None

    @field_validator("pollutant_avg", "pollutant_min", "pollutant_max", mode="before")
    @classmethod
    def parse_numeric(cls, v):
        if v is None or v == "" or v == "NA" or v == "N/A":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class StationMetadata(BaseModel):
    """Metadata representing a physical CAAQMS monitoring station."""
    station_id: str
    station_name: str
    city: str
    latitude: float
    longitude: float
    station_type: str = "Urban / Background"
