from __future__ import annotations
from typing import Literal
import uuid
from pydantic import BaseModel, Field


class StartTransferSessionDTO(BaseModel):
    """Data to initiate a file transfer session."""

    transfer_type: Literal["video", "image", "panorama"] = Field(
        ..., description="The type of file being uploaded."
    )
    size: int = Field(
        ..., description="Number of image if image else the total size of the file'"
    )

    name: str = Field(..., description="The name of the tourview")


class StartTransferResponseDTO(BaseModel):
    """Response after starting a transfer session."""

    session_id: uuid.UUID = Field(
        ..., description="The unique ID for this transfer session."
    )
