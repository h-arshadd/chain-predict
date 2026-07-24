"""
main.py
-------
FastAPI app entrypoint. Run with:

    uvicorn api.main:app --reload

from the repo root (same level as crypto_pipeline/ and api/), so that
`from crypto_pipeline...` and `from api...` imports both resolve. Same
convention as the pipeline's own run_*.bat scripts -- run from repo root.

Routers are added one per module (per spec: "each module should own its
routes"). wallets, executions, sentiment, and strategies exist so far --
backtests/models/dashboard land here the same way as we build them, no
restructuring needed.

No /api/users, /api/auth -- intentionally absent per instructions, this
is a single-operator tool with no login.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routers import wallets, executions, sentiment, strategies

app = FastAPI(title="Trading Platform API")

# Vite's default dev server port. Adjust/extend if the frontend runs
# elsewhere (e.g. add your deployed frontend origin here too).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Keep error shape consistent with the rest of the app ({"detail": ...})
    # instead of FastAPI's default verbose validation body.
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(wallets.router)
app.include_router(executions.router)
app.include_router(sentiment.router)
app.include_router(strategies.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}