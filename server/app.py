# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Orbital Anomaly OpenEnv Environment.

Exposes the simulator over HTTP and WebSocket endpoints via OpenEnv's
standard server interface.  A custom /reset override passes the optional
``task_id`` field from the request body directly to the environment so
that the Phase-2 grader can target specific benchmark tasks by name.
"""

from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from models import (
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
)
from server.orbital_anomaly_openenv_environment import (
    OrbitalAnomalyOpenenvEnvironment,
)

# ── Build the OpenEnv FastAPI app ─────────────────────────────────────────────

app = create_app(
    OrbitalAnomalyOpenenvEnvironment,
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
    env_name="orbital_anomaly_openenv",
    max_concurrent_envs=8,
)


# ── Custom /reset that forwards task_id ───────────────────────────────────────
# We replace the auto-generated reset with one that reads ``task_id`` from
# the JSON body and passes it to env.reset().  If the body is empty or
# task_id is omitted the environment cycles tasks deterministically.

@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id: Optional[str] = body.get("task_id") if isinstance(body, dict) else None

    # Obtain (or create) the shared environment instance from the app state
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env

    obs = env.reset(task_id=task_id)

    return JSONResponse(
        content={
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }
    )


# ── Landing page ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orbital Anomaly OpenEnv</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #0b1020, #111827);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .card {
                width: 900px;
                max-width: 92%;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.35);
            }
            h1 { margin-top: 0; font-size: 42px; }
            p  { color: #cbd5e1; line-height: 1.7; font-size: 18px; }
            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
                margin-top: 28px;
            }
            .btn {
                display: block;
                text-decoration: none;
                color: white;
                padding: 18px;
                border-radius: 14px;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                transition: 0.2s ease;
                font-size: 17px;
            }
            .btn:hover { transform: translateY(-2px); background: rgba(255,255,255,0.14); }
            .footer { margin-top: 28px; color: #94a3b8; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛰️ Orbital Anomaly OpenEnv</h1>
            <p>
                A mission-control benchmark for diagnosing cascading spacecraft
                subsystem failures and applying multi-step recovery policies
                across power, thermal, communication, and payload systems.
            </p>
            <div class="grid">
                <a class="btn" href="/docs">📘 Interactive API Docs</a>
                <a class="btn" href="/schema">🧩 JSON Schema</a>
                <a class="btn" href="/state">📡 Live Environment State</a>
                <a class="btn" href="/openapi.json">⚙️ OpenAPI Spec</a>
            </div>
            <div class="footer">
                Built with OpenEnv · FastAPI · Hugging Face Spaces
                | Tasks: easy · medium · hard
            </div>
        </div>
    </body>
    </html>
    """


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()