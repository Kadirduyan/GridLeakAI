"""API'ye gelen verilerin alanlarını ve geçerli değerlerini tanımlar."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NoteInput(InputModel):
    note: str = Field(max_length=5000)


class CapacityInput(InputModel):
    capacity: int = Field(strict=True, ge=1, le=1000)


class InspectionStatus(str, Enum):
    planned = "Planlandı"
    ongoing = "İnceleme sürüyor"
    confirmed = "Kaçak doğrulandı"
    cleared = "Kaçak bulunmadı"
    followup = "Ek inceleme gerekli"


class ResultInput(NoteInput):
    status: InspectionStatus
