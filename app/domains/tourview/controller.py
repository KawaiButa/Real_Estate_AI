from __future__ import annotations
from typing import List, Union
import uuid
from litestar import Controller, Response, get, post, status_codes
from litestar.di import Provide
from litestar.params import Body
from litestar.datastructures import UploadFile
from domains.tourview.service import TourviewService, provide_tourview_service
from domains.tourview.dtos import (
    StartTransferSessionDTO,
    StartTransferResponseDTO,
    FinalizeTransferResponseDTO,
)
from database.models.tourview import Tourview
from litestar.background_tasks import BackgroundTask

class TourviewController(Controller):
    path = "/properties/{property_id:uuid}/tourview"
    dependencies = {"service": Provide(provide_tourview_service)}

    @get(
        no_auth=True,
    )
    async def get_tourview_image(
        self, service: TourviewService, property_id: uuid.UUID
    ) -> List[Tourview]:
        """
        1. Endpoint to get the tourview image of the Property.
        """
        return await service.get_tourview_for_property(property_id)

    @post("/images")
    async def upload_tourview_images(
        self,
        service: TourviewService,
        property_id: uuid.UUID,
        data: list[UploadFile] = Body(
            description="A list of images to be stitched into a tourview."
        ),
    ) -> Tourview:
        """
        2. Endpoint to upload a list of images to create the Tourview image.
        """
        return await service.create_from_images(property_id, data)

    @post("/transfer/start", no_auth=True)
    async def start_transfer_session(
        self,
        service: TourviewService,
        property_id: uuid.UUID,
        data: StartTransferSessionDTO,
    ) -> StartTransferResponseDTO:
        """
        3. Endpoint to start a session for transferring a video or multiple images.
        """
        session_id = await service.start_transfer_session(property_id, data)
        return StartTransferResponseDTO(session_id=session_id)

    @post(
        "/transfer/{session_id:uuid}/chunk",
        path_override="/tourview/transfer/{session_id:uuid}/chunk",  # Override class path
        no_auth=True,
    )
    async def upload_chunk(
        self,
        service: TourviewService,
        session_id: uuid.UUID,
        body: bytes,
    ) -> str:
        """
        4 & 5. Endpoint to send a video or panorama chunk.
        This single endpoint handles both based on the session type.
        """
        await service.save_chunk(session_id, body)
        return "Your chunk is save successfully"

    @post(
        "/transfer/{session_id:uuid}/finalize",
        # path_override="/tourview/transfer/{session_id:uuid}/finalize",  # Override class path
        no_auth=True,
    )
    async def finalize_transfer(
        self,
        property_id: uuid.UUID,
        service: TourviewService,
        session_id: uuid.UUID,
        
    ) -> str:
        """
        Extra Endpoint: Finalizes a chunked transfer after all chunks are uploaded.
        This is necessary to trigger the processing of video/panorama files.
        """
        result, name = await service.finalize_transfer(session_id)
        background_task = BackgroundTask(self.create_tourview, result, property_id, name, service)
        return Response("Transfer completed successfully.", background=background_task)
    async def create_tourview(self, path: Union[str, list[str]], property_id: uuid.UUID, name: str, service: TourviewService):
        if path is str:
            service.stitch_video(path, property_id, name)
        else:
            service.stitch_images(path, property_id, name)