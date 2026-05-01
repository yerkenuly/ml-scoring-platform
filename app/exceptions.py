from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.detail


class NotFoundError(AppError):
    status_code = 404
    detail = "Resource not found"


class ValidationError(AppError):
    status_code = 422
    detail = "Validation error"


class ConflictError(AppError):
    status_code = 409
    detail = "Resource already exists"


class UnauthorizedError(AppError):
    status_code = 401
    detail = "Invalid or missing API key"


class NoStableModelError(AppError):
    status_code = 422
    detail = "No stable model found among candidates"


class ModelNotReadyError(AppError):
    status_code = 503
    detail = "No production model loaded"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
