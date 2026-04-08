# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Orbital Anomaly OpenEnv Environment V2.

Exposes the V2 simulator over HTTP and WebSocket endpoints via OpenEnv's
standard server interface. The custom /reset override passes the optional
``task_id`` from the request body to the environment so the Phase-2 grader
can target specific benchmark tasks by name.
"""

from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

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

@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_id: Optional[str] = body.get("task_id") if isinstance(body, dict) else None
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.reset(task_id=task_id)

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


# ── Landing page ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orbital Anomaly OpenEnv V2</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: radial-gradient(ellipse at 20% 20%, #0d1b3e, #060d1c 70%);
                color: #e2e8f0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .card {
                width: 960px;
                max-width: 100%;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(100,180,255,0.15);
                border-radius: 24px;
                padding: 48px;
                box-shadow: 0 24px 80px rgba(0,0,0,0.5);
            }
            h1 { font-size: 38px; font-weight: 700; margin-bottom: 8px; }
            .tag {
                display: inline-block;
                background: rgba(59,130,246,0.25);
                border: 1px solid rgba(59,130,246,0.4);
                color: #93c5fd;
                font-size: 12px;
                padding: 3px 10px;
                border-radius: 20px;
                margin-bottom: 20px;
            }
            p { color: #94a3b8; line-height: 1.75; font-size: 16px; margin-bottom: 24px; }
            .subsystems {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom: 28px;
            }
            .sub {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 14px 16px;
                font-size: 13px;
            }
            .sub strong { color: #7dd3fc; display: block; margin-bottom: 4px; }
            .sub span { color: #94a3b8; }
            .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
            .btn {
                display: block;
                text-decoration: none;
                color: #e2e8f0;
                padding: 16px 20px;
                border-radius: 12px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.1);
                transition: 0.2s ease;
                font-size: 15px;
            }
            .btn:hover { background: rgba(59,130,246,0.15); border-color: rgba(59,130,246,0.4); }
            .footer { margin-top: 24px; color: #475569; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛰️ Orbital Anomaly OpenEnv</h1>
            <span class="tag">Version 2.0 · Digital Twin</span>
            <p>
                A mission-control benchmark featuring a full spacecraft digital twin with
                EPS power-balance physics, ADCS attitude dynamics, multi-zone thermal
                propagation, RF communications chain modeling, orbital context (eclipse,
                ground station windows, science observation windows), latent root-cause
                faults with delayed cascading failures, and partial observability.
            </p>
            <div class="subsystems">
                <div class="sub"><strong>⚡ EPS</strong><span>Power balance, SOC, bus voltage, panel health, MPPT faults</span></div>
                <div class="sub"><strong>🧭 ADCS</strong><span>Attitude error, sun alignment, wheel saturation, gyro drift</span></div>
                <div class="sub"><strong>🌡️ Thermal</strong><span>3-zone propagation: battery / payload / avionics</span></div>
                <div class="sub"><strong>📡 Comms</strong><span>BER, packet loss, uplink margin, antenna pointing</span></div>
                <div class="sub"><strong>🌍 Orbital</strong><span>Eclipse, GS windows, science windows, radiation belt</span></div>
                <div class="sub"><strong>🔥 Fault Graph</strong><span>13 latent root faults with delayed cascading effects</span></div>
            </div>
            <div class="grid">
                <a class="btn" href="/docs">📘 Interactive API Docs</a>
                <a class="btn" href="/schema">🧩 JSON Schema</a>
                <a class="btn" href="/state">📡 Live Environment State</a>
                <a class="btn" href="/openapi.json">⚙️ OpenAPI Spec</a>
            </div>
            <div class="footer">
                Built with OpenEnv · FastAPI · Hugging Face Spaces
                &nbsp;|&nbsp; Tasks: easy · medium · hard
                &nbsp;|&nbsp; Actions: 6 &nbsp;|&nbsp; Subsystems: 5 &nbsp;|&nbsp; Latent faults: 13
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