import argparse
import json
import logging

from typing import Optional

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp
import uvicorn

from langport.constants import ErrorCode
from fastapi.exceptions import RequestValidationError
from langport.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingsRequest,
)
from langport.routers.gateway.common import AppSettings, create_error_response
from langport.routers.gateway.openai_compatible import api_chat_completions, api_completions, api_embeddings, api_models

logger = logging.getLogger(__name__)

app = fastapi.FastAPI(debug=False)

class BaseAuthorizationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, sk:str, dispatch: Optional[DispatchFunction] = None) -> None:
        super().__init__(app, dispatch)
        self.sk = sk

    async def dispatch(self, request, call_next):
        authorization = request.headers.get("Authorization")
        if request.url.path not in ["/docs","/redoc","/openapi.json"] and (
            not authorization 
            or authorization.split(" ")[0].lower() != "bearer" 
            or authorization.split(" ")[1] != self.sk
        ):
            return JSONResponse(
                status_code=401,
                content={"msg":"Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


class RedirectModelMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp,  redirect_rules:list, dispatch: Optional[DispatchFunction] = None) -> None:
        super().__init__(app, dispatch)
        self.redirect_rules = redirect_rules
        self.receive_ = None

    async def dispatch(self, request, call_next):
        if "content-type" not in request.headers or "application/json" not in request.headers["content-type"]:
            return await call_next(request)
        
        try:
            await self.set_body(request)
            data = await request.json()
            if "model" in data:
                for rule in self.redirect_rules:
                    from_model_name, to_model_name = rule.split(":")
                    if data["model"] == from_model_name:
                        data["model"] = to_model_name
                        self.receive_['body'] = json.dumps(data).encode("utf-8")
                        break
        except Exception as e:
            logger.error(f"RedirectModelMiddleware: {e}")
        return await call_next(request)
    
    async def set_body(self, request):
        self.receive_ = await request._receive()
        async def receive():
            return self.receive_
        request._receive = receive


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


@app.get("/v1/models")
async def models():
    return await api_models(app.app_settings)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await api_chat_completions(app.app_settings, request)

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    return await api_completions(app.app_settings, request)


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    return await api_embeddings(app.app_settings, request)


if __name__ in ["__main__", "langport.service.gateway.openai_api"]:
    parser = argparse.ArgumentParser(
        description="Langport openai-compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--sk", type=str, default=None, help="security key")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument("--redirect", action="append", required=False, help="redirect model_name to another model_name, e.g. --redirect model_name1:model_name2")
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    if args.redirect is not None:
        app.add_middleware(
            RedirectModelMiddleware,
            redirect_rules=args.redirect,
        )
    if args.sk is not None:
        app.add_middleware(
            BaseAuthorizationMiddleware,
            sk=args.sk,
        )
    app.app_settings = AppSettings(controller_address=args.controller_address)

    logger.debug(f"==== args ====\n{args}")

    # don't delete this line, otherwise the middleware won't work with reload==True
    if __name__ == "__main__":
        uvicorn.run(
            "langport.service.gateway.openai_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=True,
        )
