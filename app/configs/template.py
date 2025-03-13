from pathlib import Path
from litestar.template import TemplateConfig
from litestar.contrib.jinja import JinjaTemplateEngine
template_config = TemplateConfig(
    directory=Path("templates"),
    engine=JinjaTemplateEngine,
)
