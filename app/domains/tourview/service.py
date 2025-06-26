from collections.abc import AsyncGenerator
import os
import uuid
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from database.models.tourview import Tourview
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
from litestar.datastructures import UploadFile
from domains.image.service import ImageService
from domains.supabase.service import SupabaseService
from domains.tourview.dtos import CreatePanoramaDTO
from litestar.exceptions import InternalServerException


class TourviewRepository(SQLAlchemyAsyncRepository[Tourview]):
    model_type = Tourview


class TourviewService(SQLAlchemyAsyncRepositoryService[Tourview]):
    repository_type = TourviewRepository
    supabase_service: SupabaseService = SupabaseService()

    async def create_tourview(self, property_id: uuid.UUID, data: CreatePanoramaDTO) -> Tourview:
        try:
            image = self.stitch_panorama_with_caps_from_uploads(
                data.images[0], data.images[1], data.images[2]
            )
            image_service = ImageService(session=self.repository.session)
            image = await image_service.create(
                data={
                    "url": await self.supabase_service.upload_file(image),
                    "model_type": None,
                    "model_id": None,
                }
            )
            tourview = await self.create(
                data={
                    "image_id": image.id,
                    "property_id": property_id,
                }
            )
            return tourview
        except:
            self.repository.session.rollback()
            raise InternalServerException(
                "There is something wrong with the tourview service. Please try again later"
            )
        finally:
            self.repository.session.commit()

    def stitch_panorama_with_caps_from_uploads(
        self,
        panorama_file: UploadFile,
        up_file: UploadFile,
        down_file: UploadFile,
    ) -> Image:
        pano = Image.open(panorama_file.file)
        up = Image.open(up_file.file)
        down = Image.open(down_file.file)

        target_width = pano.width
        if up.width != target_width:
            up = up.resize((target_width, int(up.height * target_width / up.width)))
        if down.width != target_width:
            down = down.resize(
                (target_width, int(down.height * target_width / down.width))
            )

        total_height = up.height + pano.height + down.height
        combined = Image.new("RGB", (target_width, total_height))
        combined.paste(up, (0, 0))
        combined.paste(pano, (0, up.height))
        combined.paste(down, (0, up.height + pano.height))
        return combined


async def provide_tourview_service(
    db_session: AsyncSession,
) -> AsyncGenerator[TourviewService]:
    async with TourviewService.new(session=db_session) as service:
        yield service
