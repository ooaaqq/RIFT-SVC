"""Production ASGI launcher."""

from __future__ import annotations

import uvicorn

from rift_web.app import create_app
from rift_web.config import Settings


def main() -> None:
    settings = Settings.from_environment()
    uvicorn.run(
        create_app(settings),
        host=settings.listen_host,
        port=settings.listen_port,
        access_log=True,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
