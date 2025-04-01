from litestar import Controller, get
from litestar.di import Provide

from database.models.property_type import PropertyType
from domains.property_types.service import (
    PropertyTypeService,
    provide_property_type_service,
)


class PropertyTypeController(Controller):
    path = "/properties/types"
    dependencies = {"property_type_service": Provide(provide_property_type_service)}

    @get(
        "/",
        no_auth=True,
    )
    async def get_property_types(
        self, property_type_service: PropertyTypeService
    ) -> list[PropertyType]:
        return await property_type_service.list()
