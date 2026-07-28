"""
main.py
-------
FastAPI app entrypoint. Run with:

    uvicorn api.main:app --reload

from the repo root (same level as crypto_pipeline/ and api/), so that
`from crypto_pipeline...` and `from api...` imports both resolve. Same
convention as the pipeline's own run_*.bat scripts -- run from repo root.

Routers are added one per module (per spec: "each module should own its
routes"). wallets, executions, sentiment, strategies, ml, dashboard,
backtests, and simulator exist so far.

Every request (any method/router, including 404s) is logged to
public.request_log via request_logging_middleware below -- see
core/request_log.py for the schema. Not request/response bodies, only
method/path/query/status/duration/client_ip (see that module's
docstring for why bodies are excluded).

No /api/users, /api/auth -- intentionally absent per instructions, this
is a single-operator tool with no login.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import time

from api.routers import wallets, executions, sentiment, strategies, ml, dashboard, backtests, simulator, playbook
from api.core.request_log import log_request

app = FastAPI(title="Trading Platform API")

# Vite's default dev server port. Adjust/extend if the frontend runs
# elsewhere (e.g. add your deployed frontend origin here too).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Logs every request (any method, any router, including 404s and
    validation errors) to public.request_log -- see core/request_log.py
    for the schema and what's deliberately NOT logged (request/response
    bodies, since some POST/PUT bodies here carry real API secrets).

    Runs the actual request first so status_code/duration are real. The
    DB write itself is best-effort (see log_request) so a logging hiccup
    never turns a working request into a failed one.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    log_request(
        method=request.method,
        path=request.url.path,
        query_string=str(request.url.query) if request.url.query else None,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
        client_ip=request.client.host if request.client else None,
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Keep error shape consistent with the rest of the app ({"detail": ...})
    # instead of FastAPI's default verbose validation body.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(wallets.router)
app.include_router(executions.router)
app.include_router(sentiment.router)
app.include_router(strategies.router)
app.include_router(ml.router)
app.include_router(dashboard.router)
app.include_router(backtests.router)
app.include_router(simulator.router)
app.include_router(playbook.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}