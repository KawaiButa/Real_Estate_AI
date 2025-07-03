from collections.abc import AsyncGenerator
from pathlib import Path
import shutil
import cv2
from typing import List, Optional, Tuple, Union
import uuid
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
import numpy as np
from domains.tourview.dtos import (
    StartTransferSessionDTO,
)
from database.models.tourview import Tourview
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
from domains.supabase.service import SupabaseService
import aiofiles

from litestar.exceptions import (
    NotFoundException,
    ClientException,
    InternalServerException,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


class TourviewRepository(SQLAlchemyAsyncRepository[Tourview]):
    model_type = Tourview


transfer_sessions = {}

UPLOAD_DIR = Path("uploads/tourview")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TourviewService(SQLAlchemyAsyncRepositoryService[Tourview]):
    repository_type = TourviewRepository
    supabase_service: SupabaseService = SupabaseService()

    async def get_tourview_for_property(self, property_id: uuid.UUID) -> Tourview:
        """Retrieves the tourview associated with a property."""
        await self._get_property(property_id)  # Ensure property exists

        query = select(Tourview).where(Tourview.property_id == property_id)
        result = await self.repository.session.execute(query)
        tourview = result.scalar_one_or_none()

        if not tourview:
            raise NotFoundException(f"No Tourview found for property {property_id}.")

        return tourview

    async def create_from_images(
        self, property_id: uuid.UUID, images: list
    ) -> Tourview:
        """
        Creates a tourview from a list of uploaded images by stitching them.
        """
        prop = await self._get_property(property_id)

        # 1. Save uploaded images temporarily
        temp_dir = UPLOAD_DIR / str(uuid.uuid4())
        temp_dir.mkdir()

        image_paths = []
        for image_file in images:
            file_path = temp_dir / image_file.filename
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await image_file.read())
            image_paths.append(file_path)

        # 2. Stitch the images (YOUR IMPLEMENTATION GOES HERE)
        try:
            final_image_path = await self.stitch_images(image_paths)
        except Exception as e:
            # Clean up on failure
            shutil.rmtree(temp_dir)
            raise ClientException(f"Failed to process images: {e}") from e

        # 3. Create database records
        # NOTE: final_image_path should be a public URL after moving it to a public storage (e.g., S3)
        new_image = Image(
            url=str(final_image_path),  # This should be a URL, not a local path in prod
            model_id=prop.id,
            model_type="property_tourview",
        )

        new_tourview = Tourview(
            name=f"Tour View for {prop.title}", property_id=prop.id, image=new_image
        )

        self.repository.session.add(new_tourview)
        await self.repository.session.commit()
        await self.repository.session.refresh(new_tourview)

        # 4. Clean up temporary files
        shutil.rmtree(temp_dir)

        return new_tourview

    async def start_transfer_session(
        self, property_id: uuid.UUID, data: StartTransferSessionDTO
    ) -> uuid.UUID:
        """Starts a new chunked transfer session."""
        # await self._get_property(property_id)  # Ensure property exists
        session_id = uuid.uuid4()
        session_dir = UPLOAD_DIR / str(session_id)
        session_dir.mkdir()

        transfer_sessions[session_id] = {
            "property_id": property_id,
            "type": data.transfer_type,
            "path": session_dir,
            "file_count": 0,
            "expected_files": (data.size if data.transfer_type == "image" else None),
            "current_size": 0 if data.transfer_type != "image" else None,
            "size": data.size if data.transfer_type != "image" else None,
            "name": data.name,
        }
        return session_id

    async def save_chunk(self, session_id: uuid.UUID, chunk: bytes) -> Optional[str]:
        """Saves a single chunk for a given session."""
        try:
            session = transfer_sessions.get(session_id)
            if not session:
                raise NotFoundException("Invalid transfer session ID.")
            chunk_size = len(chunk)
            file_path = session["path"] / f"{session["file_count"] + 1}"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(chunk)

            session["file_count"] += 1
            session["current_size"] += chunk_size
            return file_path
        except Exception as e:
            print(e)
            raise InternalServerException()

    async def finalize_transfer(
        self, session_id: uuid.UUID
    ) -> Tuple[Union[str, List[str]], str]:
        """
        Finalizes a transfer, processes files, and creates the Tourview.
        This endpoint is crucial for video and panorama uploads.
        """
        session = transfer_sessions.get(session_id)
        if not session:
            raise NotFoundException("Invalid transfer session ID.")
        file_paths = sorted(list(session["path"].glob("*")))
        if not file_paths:
            raise ClientException("No files found in this transfer session.")
        try:
            if session["type"] in ["video", "panorama"]:
                final_processed_path = await self.reassemble_and_process_chunks(
                    file_paths, session["type"]
                )
                return final_processed_path, session["name"]
            elif session["type"] == "image":
                return file_paths, session["name"]
            else:
                raise ClientException("Unsupported transfer type for finalization.")
        except Exception as e:
            shutil.rmtree(session["path"])
            raise ClientException(f"Processing failed: {e}") from e
        finally:
            del transfer_sessions[session_id]

    async def reassemble_and_process_chunks(
        self, chunk_paths: list[Path], file_type: str
    ) -> Path:
        """
        Reassembles video/panorama chunks into a single file and returns the final file path.
        """
        print(f"Reassembling {len(chunk_paths)} chunks for a {file_type}...")

        final_file = (
            UPLOAD_DIR
            / f"final_{uuid.uuid4()}.{ 'mp4' if file_type == 'video' else 'jpg'}"
        )

        async with aiofiles.open(final_file, "wb") as f_out:
            for chunk_path in chunk_paths:
                async with aiofiles.open(chunk_path, "rb") as f_in:
                    while True:
                        chunk_data = await f_in.read(1024 * 1024)  # read in 1MB blocks
                        if not chunk_data:
                            break
                        await f_out.write(chunk_data)

        print(f"Reassembly complete. Final file at {final_file}")
        return final_file

    def extract_and_select_frames(
        self, video_path, min_movement=5, max_movement=50, step=5
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        selected_frames = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                # Calculate optical flow or feature movement
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                movement = np.linalg.norm(flow, axis=2).mean()
                print(f"Frame {frame_idx}: avg movement={movement:.2f}")

                if min_movement < movement < max_movement:
                    selected_frames.append(frame)
            else:
                # Always select the first frame
                selected_frames.append(frame)

            prev_gray = gray
            frame_idx += 1

        cap.release()
        return selected_frames

    def stitch_video(path: str, property_id: uuid.UUID, name: str) -> Tourview:
        pass

    def stich_images(paths: list[str], property_id: uuid.UUID, name: str) -> Tourview:
        pass


async def provide_tourview_service(
    db_session: AsyncSession,
) -> AsyncGenerator[TourviewService]:
    async with TourviewService.new(session=db_session) as service:
        yield service
