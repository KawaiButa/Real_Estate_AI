from __future__ import annotations
import asyncio
import threading
import time
from typing import Any, List, Union
import uuid
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.params import Body
from litestar.datastructures import UploadFile
from database.models.user import User
from domains.notification.service import NotificationService
from domains.tourview.service import TourviewService, provide_tourview_service
from domains.tourview.dtos import (
    StartTransferSessionDTO,
    StartTransferResponseDTO,
)
from database.models.tourview import Tourview
from litestar.background_tasks import BackgroundTask, BackgroundTasks
from litestar.security.jwt import Token


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
        # no_auth=True,
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
        # no_auth=True,
    )
    async def finalize_transfer(
        self,
        property_id: uuid.UUID,
        request: Request[User, Token, Any],
        service: TourviewService,
        session_id: uuid.UUID,
    ) -> Response:
        """
        Extra Endpoint: Finalizes a chunked transfer after all chunks are uploaded.
        This is necessary to trigger the processing of video/panorama files.
        """
        result, name = await service.finalize_transfer(session_id)
        return Response(
            "Transfer completed successfully.",
            background=BackgroundTasks(
                [
                    BackgroundTask(
                        self._launch_create_tourview,
                        request.user,
                        result,
                        property_id,
                        name,
                        service,
                    )
                ]
            ),
        )

    def notify_partner(self, user: User, tourview: Union[Tourview, None]):
        if not user.device_token:
            return
        notify_service = NotificationService()
        title = "Tourview Fail" if tourview is None else "Tourview Complete"
        body = (
            f"Fail when trying to create your tourview."
            if tourview is None
            else f"Your tourview for {tourview.name} is complete. Try it right away!!!"
        )
        notify_service.send_to_token(
            token=user.device_token,
            title=title,
            body=body,
            data=(
                {
                    "type": "tourview",
                    "id": str(tourview.id),
                    "url": str(tourview.image.url),
                }
                if tourview
                else None
            ),
        )

    async def create_tourview(
        self,
        user: User,
        path: Union[str, list[str]],
        property_id: uuid.UUID,
        name: str,
        service: TourviewService,
    ) -> None:
        if isinstance(path, str):
            tourview = await service.stitch_video(path, property_id, name)
        else:
            tourview = await service.stitch_images(path, property_id, name)
        self.notify_partner(user, tourview)
    
    def _launch_create_tourview(
        self,
        user: User,
        path: Union[str, list[str]],
        property_id: uuid.UUID,
        name: str,
        service: TourviewService,
    ) -> None:
        threading.Thread(
            target=asyncio.run,
            args=(self.create_tourview(user, path, property_id, name, service),),
            daemon=True,
        ).start()