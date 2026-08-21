from __future__ import annotations

import uvicorn

from .settings import settings


def main() -> None:
    uvicorn.run(
        "app.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
