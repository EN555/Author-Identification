import logging
import uvicorn

from datasetapi.controllers.http_controller import app
from datasetapi.models.config import config as settings

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting dataset-api Service")
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()