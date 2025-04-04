from litestar import Controller, get
from litestar.di import Provide
from database.models.banner import Banner
from domains.banner.service import BannerService, provide_banner_service


class BannerController(Controller):

    path = "/banners"
    tags = ["Banner"]
    dependencies = {"banner_service": Provide(provide_banner_service)}

    @get(no_auth=True)
    async def get_banners(self, banner_service: BannerService) -> list[Banner]:
        return await banner_service.list()
