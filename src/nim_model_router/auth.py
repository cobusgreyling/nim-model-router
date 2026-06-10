from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class RouterAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, router_api_key: str) -> None:
        super().__init__(app)
        self.router_api_key = router_api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.router_api_key:
            return await call_next(request)

        if request.url.path in {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        provided = ""
        if auth_header.startswith("Bearer "):
            provided = auth_header.removeprefix("Bearer ").strip()
        elif api_key_header:
            provided = api_key_header.strip()

        if provided != self.router_api_key:
            return JSONResponse(status_code=401, content={"detail": "invalid router API key"})

        return await call_next(request)
