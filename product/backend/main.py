import logging

import uvicorn
from pipeline.routes import app

from product.backend.models.config import config as settings

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting backend Service")
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug_mode,
        workers=2,
    )


if __name__ == "__main__":
    main()
