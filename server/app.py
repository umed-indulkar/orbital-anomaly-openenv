# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations
from typing import Optional
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

from models import OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation
from server.orbital_anomaly_openenv_environment import OrbitalAnomalyOpenenvEnvironment
from inference import compute_fault_beliefs, dominant_subsystem, top_faults_str, mission_commander_decide

app = create_app(
    OrbitalAnomalyOpenenvEnvironment,
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
    env_name="orbital_anomaly_openenv",
    max_concurrent_envs=8,
)


@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id: Optional[str] = body.get("task_id") if isinstance(body, dict) else None
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.reset(task_id=task_id)
    meta     = obs.metadata or {}
    beliefs  = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    dom      = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
    top3     = top_faults_str(beliefs, 3)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward":      obs.reward,
        "done":        obs.done,
        "world_model": {
            "dominant_subsystem": dom,
            "top_faults":         top3,
            "fault_beliefs":      beliefs,
            "phase":              meta.get("phase", 0),
            "phase_step":         meta.get("phase_step", 0),
        },
    })


@app.post("/step", include_in_schema=False)
async def step_with_rationale(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    action_type: str = (body.get("action_type", "noop") if isinstance(body, dict) else "noop")
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action_type))
    meta     = obs.metadata or {}
    beliefs  = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    dom      = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
    top3     = top_faults_str(beliefs, 3)
    action_rec, rationale, specialist_recs = mission_commander_decide(obs)
    specialists = {
        name: {"action": rec[0], "confidence": round(rec[1], 3), "reason": rec[2]}
        for name, rec in specialist_recs.items()
    }
    return JSONResponse(content={
        "observation":  obs.model_dump(),
        "reward":       obs.reward,
        "done":         obs.done,
        "action_taken": action_type,
        "agent_decision": {
            "recommended_action": action_rec,
            "rationale":          rationale,
            "specialists":        specialists,
        },
        "world_model": {
            "dominant_subsystem": dom,
            "top_faults":         top3,
            "fault_beliefs":      beliefs,
            "phase":              meta.get("phase", 0),
            "phase_step":         meta.get("phase_step", 0),
        },
    })


@app.get("/decide")
async def decide(request: Request) -> JSONResponse:
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs     = env._get_observation(reward=0.5, done=False)
    action, rationale, recs = mission_commander_decide(obs)
    meta    = obs.metadata or {}
    beliefs = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    return JSONResponse(content={
        "recommended_action": action,
        "rationale":          rationale,
        "specialists": {
            name: {"action": r[0], "confidence": round(r[1], 3), "reason": r[2]}
            for name, r in recs.items()
        },
        "world_model": {
            "dominant_subsystem": meta.get("dominant_subsystem", dominant_subsystem(beliefs)),
            "top_faults":         top_faults_str(beliefs, 3),
        },
    })


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🛰️ Orbital Anomaly — Mission Control</title>
<style>
  :root {
    --bg:#060d1c;--panel:#0d1829;--border:rgba(99,179,237,0.15);
    --text:#e2e8f0;--muted:#64748b;--accent:#3b82f6;
    --green:#10b981;--yellow:#f59e0b;--red:#ef4444;--purple:#8b5cf6;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;font-size:13px;min-height:100vh}
  header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;background:rgba(13,24,41,0.95);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
  .logo{font-size:18px;font-weight:700;letter-spacing:.5px}
  .logo span{color:var(--accent)}
  .status-pill{display:flex;align-items:center;gap:8px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:20px;padding:4px 12px;font-size:12px}
  .status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .grid{display:grid;grid-template-columns:280px 1fr 320px;gap:16px;padding:16px;max-width:1400px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}
  .panel-title{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:6px}
  .panel-title::before{content:'';display:block;width:3px;height:12px;background:var(--accent);border-radius:2px}
  .gauge-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .gauge{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:10px 12px}
  .gauge-label{font-size:10px;color:var(--muted);margin-bottom:4px}
  .gauge-value{font-size:20px;font-weight:700;margin-bottom:6px}
  .gauge-bar{height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden}
  .gauge-fill{height:100%;border-radius:2px;transition:width .6s ease,background .6s}
  .ok{color:var(--green)} .warn{color:var(--yellow)} .crit{color:var(--red)}
  .ok-fill{background:var(--green)} .warn-fill{background:var(--yellow)} .crit-fill{background:var(--red)}
  .badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
  .badge{font-size:10px;padding:2px 8px;border-radius:10px;border:1px solid;letter-spacing:.5px}
  .badge-green{color:var(--green);border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)}
  .badge-yellow{color:var(--yellow);border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.1)}
  .badge-red{color:var(--red);border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.1)}
  .badge-blue{color:var(--accent);border-color:rgba(59,130,246,.4);background:rgba(59,130,246,.1)}
  .badge-purple{color:var(--purple);border-color:rgba(139,92,246,.4);background:rgba(139,92,246,.1)}
  .fault-list{display:flex;flex-direction:column;gap:5px}
  .fault-row{display:grid;grid-template-columns:160px 1fr 38px;align-items:center;gap:8px}
  .fault-name{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .fault-bar-bg{height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden}
  .fault-bar-fill{height:100%;border-radius:3px;transition:width .5s ease,background .5s}
  .fault-pct{font-size:11px;text-align:right}
  .action-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .action-btn{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:10px 8px;color:var(--text);cursor:pointer;font-family:inherit;font-size:12px;transition:all .15s ease;text-align:center;line-height:1.3}
  .action-btn:hover{background:rgba(59,130,246,.15);border-color:var(--accent)}
  .action-btn:active{transform:scale(.97)}
  .action-btn .btn-icon{font-size:18px;display:block;margin-bottom:2px}
  .action-btn .btn-label{font-size:10px;color:var(--muted)}
  .decision-box{background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.2);border-radius:8px;padding:12px;margin-top:10px}
  .decision-action{font-size:15px;font-weight:700;color:var(--accent);margin-bottom:4px}
  .decision-rationale{font-size:11px;color:#94a3b8;line-height:1.5}
  .specialist-list{margin-top:10px;display:flex;flex-direction:column;gap:4px}
  .specialist-row{display:flex;justify-content:space-between;align-items:center;font-size:11px;padding:4px 8px;background:rgba(255,255,255,.03);border-radius:4px}
  .spec-name{color:var(--muted)} .spec-action{color:var(--text)} .spec-conf{color:var(--green)}
  #reward-chart{width:100%;height:80px;background:rgba(255,255,255,.02);border-radius:6px;overflow:hidden}
  .phase-bar{display:flex;gap:4px;margin-top:10px}
  .phase-seg{flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,.08);transition:background .4s}
  .phase-seg.active{background:var(--accent)} .phase-seg.done{background:var(--green)}
  .task-selector{display:flex;gap:6px;margin-bottom:12px}
  .task-btn{flex:1;padding:6px;border-radius:6px;border:1px solid rgba(255,255,255,.1);background:transparent;color:var(--muted);cursor:pointer;font-family:inherit;font-size:11px;transition:all .15s}
  .task-btn.active{background:var(--accent);border-color:var(--accent);color:white}
  #log-feed{height:120px;overflow-y:auto;font-size:11px;line-height:1.6;color:var(--muted);padding:8px;background:rgba(0,0,0,.3);border-radius:6px;margin-top:10px}
  .log-action{color:var(--accent)} .log-reward{color:var(--green)} .log-warn{color:var(--yellow)}
  .section-spacer{margin-top:12px}
</style>
</head>
<body>
<header>
  <div class="logo">🛰️ Orbital Anomaly <span>Mission Control</span></div>
  <div style="display:flex;gap:12px;align-items:center">
    <div id="episode-score" style="font-size:12px;color:var(--muted)">Score: —</div>
    <div class="status-pill">
      <div class="status-dot"></div>
      <span id="mission-status-header">STABLE</span>
    </div>
  </div>
</header>

<div class="grid">
  <!-- LEFT: Telemetry -->
  <div>
    <div class="panel">
      <div class="panel-title">Telemetry</div>
      <div class="gauge-grid">
        <div class="gauge"><div class="gauge-label">BATTERY SOC</div><div class="gauge-value" id="g-bat">—</div><div class="gauge-bar"><div class="gauge-fill" id="gf-bat" style="width:0%"></div></div></div>
        <div class="gauge"><div class="gauge-label">SOLAR EFFIC.</div><div class="gauge-value" id="g-sol">—</div><div class="gauge-bar"><div class="gauge-fill" id="gf-sol" style="width:0%"></div></div></div>
        <div class="gauge"><div class="gauge-label">THERMAL °C</div><div class="gauge-value" id="g-temp">—</div><div class="gauge-bar"><div class="gauge-fill" id="gf-temp" style="width:0%"></div></div></div>
        <div class="gauge"><div class="gauge-label">COMMS SIGNAL</div><div class="gauge-value" id="g-comms">—</div><div class="gauge-bar"><div class="gauge-fill" id="gf-comms" style="width:0%"></div></div></div>
      </div>
      <div class="badges" id="status-badges"></div>
    </div>
    <div class="panel section-spacer">
      <div class="panel-title">Extended Mission Phase</div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px"><span>Phase 1: EPS</span><span>Phase 2: Thermal</span><span>Phase 3: Comms</span></div>
      <div class="phase-bar"><div class="phase-seg" id="ph0"></div><div class="phase-seg" id="ph1"></div><div class="phase-seg" id="ph2"></div></div>
      <div style="margin-top:6px;font-size:11px;color:var(--muted)" id="phase-label">Not started</div>
    </div>
    <div class="panel section-spacer">
      <div class="panel-title">Reward Timeline</div>
      <svg id="reward-chart" viewBox="0 0 260 80" preserveAspectRatio="none">
        <polyline id="reward-line" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points=""/>
        <line x1="0" y1="60" x2="260" y2="60" stroke="rgba(239,68,68,0.3)" stroke-width="1" stroke-dasharray="4"/>
        <text x="4" y="58" font-size="9" fill="rgba(239,68,68,0.6)">0.45</text>
      </svg>
      <div style="font-size:11px;color:var(--muted);margin-top:4px">Step: <span id="step-num">0</span> / 36 &nbsp;|&nbsp; Avg: <span id="avg-reward">—</span></div>
    </div>
  </div>

  <!-- MIDDLE: World Model + Actions -->
  <div>
    <div class="panel">
      <div class="panel-title">World Model — 13-Fault Belief State</div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:10px">Dominant: <span id="dominant-sys" style="color:var(--accent)">—</span> &nbsp;|&nbsp; Top fault: <span id="top-fault" style="color:var(--yellow)">—</span></div>
      <div class="fault-list" id="fault-list"></div>
    </div>
    <div class="panel section-spacer">
      <div class="panel-title">Take Action</div>
      <div class="task-selector">
        <button class="task-btn active" onclick="startTask('easy',this)">🟢 Easy</button>
        <button class="task-btn" onclick="startTask('medium',this)">🟡 Medium</button>
        <button class="task-btn" onclick="startTask('hard',this)">🔴 Hard</button>
      </div>
      <div class="action-grid">
        <button class="action-btn" onclick="takeAction('rotate_to_sun')"><span class="btn-icon">☀️</span>rotate_to_sun<div class="btn-label">Realign solar panels</div></button>
        <button class="action-btn" onclick="takeAction('disable_payload')"><span class="btn-icon">📦</span>disable_payload<div class="btn-label">Cut science payload</div></button>
        <button class="action-btn" onclick="takeAction('reboot_comms')"><span class="btn-icon">📡</span>reboot_comms<div class="btn-label">Restore RF link</div></button>
        <button class="action-btn" onclick="takeAction('enter_safe_mode')"><span class="btn-icon">🛡️</span>enter_safe_mode<div class="btn-label">Emergency mode</div></button>
        <button class="action-btn" onclick="takeAction('switch_power_bus')"><span class="btn-icon">🔋</span>switch_power_bus<div class="btn-label">Backup battery bus</div></button>
        <button class="action-btn" onclick="takeAction('noop')"><span class="btn-icon">⏸️</span>noop<div class="btn-label">Hold position</div></button>
      </div>
      <div id="log-feed"><div style="color:var(--muted)">Select a task to begin...</div></div>
    </div>
  </div>

  <!-- RIGHT: Agent Decision + State -->
  <div>
    <div class="panel">
      <div class="panel-title">Multi-Agent Decision</div>
      <div class="decision-box" id="agent-decision-box"><div style="color:var(--muted);font-size:12px">Reset to see agent recommendation</div></div>
      <div class="specialist-list" id="specialist-list"></div>
    </div>
    <div class="panel section-spacer">
      <div class="panel-title">Spacecraft State</div>
      <div style="display:flex;flex-direction:column;gap:6px;font-size:12px" id="state-details"><div style="color:var(--muted)">—</div></div>
    </div>
    <div class="panel section-spacer">
      <div class="panel-title">About</div>
      <div style="font-size:11px;color:var(--muted);line-height:1.7">
        <b style="color:var(--text)">Theme 3 — World Modeling</b><br>13-fault latent causal graph. Agent infers hidden fault state from observable symptoms.<br><br>
        <b style="color:var(--text)">Theme 2 — Long-Horizon</b><br>36-step Extended Mission Mode. Battery + thermal carry over between phases.<br><br>
        <b style="color:var(--text)">Multi-Agent</b><br>MissionCommander oversees EPS, Thermal, Comms specialist agents.<br><br>
        <a href="/docs" style="color:var(--accent)">📘 API Docs</a> &nbsp; <a href="/state" style="color:var(--accent)">📡 Raw State</a>
      </div>
    </div>
  </div>
</div>

<script>
const FAULT_NAMES=['mppt_stuck','panel_deployment_jam','bus_short_transient','battery_aging','reaction_wheel_saturation','gyro_drift','star_tracker_dropout','radiator_valve_stuck','heat_pipe_failure','heater_relay_latch','transponder_overheating','amplifier_degradation','antenna_gimbal_stall'];
let rewardHistory=[],stepCount=0;

function initFaultList(){
  document.getElementById('fault-list').innerHTML=FAULT_NAMES.map(n=>`
    <div class="fault-row">
      <div class="fault-name" title="${n}">${n}</div>
      <div class="fault-bar-bg"><div class="fault-bar-fill" id="fb-${n}" style="width:0%"></div></div>
      <div class="fault-pct" id="fp-${n}">—</div>
    </div>`).join('');
}

function faultColor(p){return p>0.7?'#ef4444':p>0.4?'#f59e0b':'#3b82f6'}

function updateFaults(beliefs){
  if(!beliefs)return;
  const sorted=Object.entries(beliefs).sort((a,b)=>b[1]-a[1]);
  FAULT_NAMES.forEach(n=>{
    const p=beliefs[n]||0;
    const f=document.getElementById('fb-'+n);
    const fp=document.getElementById('fp-'+n);
    if(f){f.style.width=(p*100)+'%';f.style.background=faultColor(p)}
    if(fp){fp.textContent=Math.round(p*100)+'%';fp.style.color=faultColor(p)}
  });
  if(sorted.length>0){
    document.getElementById('top-fault').textContent=
      sorted[0][0].replace(/_/g,' ')+'('+Math.round(sorted[0][1]*100)+'%)';
  }
}

function updateTelemetry(obs){
  if(!obs)return;
  const bat=(obs.battery_soc||obs.battery_level||0);
  const sol=(obs.solar_efficiency||0)*100;
  const temp=(obs.thermal_temp||0);
  const comms=(obs.comms_signal||0)*100;

  function setG(id,val,unit,lo,hi,pct){
    const cls=val<lo?'crit':val<hi?'warn':'ok';
    document.getElementById('g-'+id).textContent=val.toFixed(1)+unit;
    document.getElementById('g-'+id).className='gauge-value '+cls;
    const f=document.getElementById('gf-'+id);
    f.style.width=Math.min(100,Math.max(0,pct))+'%';
    f.className='gauge-fill '+cls+'-fill';
  }
  setG('bat',bat,'%',20,40,bat);
  setG('sol',sol,'%',30,60,sol);
  setG('comms',comms,'%',30,60,comms);

  // Thermal — higher is worse
  const tCls=temp>85?'crit':temp>70?'warn':'ok';
  document.getElementById('g-temp').textContent=temp.toFixed(1)+'C';
  document.getElementById('g-temp').className='gauge-value '+tCls;
  const tf=document.getElementById('gf-temp');
  tf.style.width=Math.min(100,(temp/120)*100)+'%';
  tf.className='gauge-fill '+tCls+'-fill';

  // Badges — safe null checks
  const sunlit=obs.sunlit!==undefined?obs.sunlit:true;
  const gs=obs.ground_station_visible!==undefined?obs.ground_station_visible:true;
  const badges=[];
  badges.push(`<span class="badge ${sunlit?'badge-green':'badge-red'}">${sunlit?'☀️ SUNLIT':'🌑 ECLIPSE'}</span>`);
  badges.push(`<span class="badge ${gs?'badge-green':'badge-yellow'}">${gs?'📡 GS VISIBLE':'📡 GS BLACKOUT'}</span>`);
  if(obs.payload_on) badges.push('<span class="badge badge-purple">🔬 PAYLOAD ON</span>');
  if(obs.safe_mode)  badges.push('<span class="badge badge-red">🛡️ SAFE MODE</span>');
  if(obs.radiation_zone) badges.push('<span class="badge badge-red">☢️ RADIATION</span>');
  document.getElementById('status-badges').innerHTML=badges.join('');

  const st=(obs.mission_status||'stable').toUpperCase();
  const stEl=document.getElementById('mission-status-header');
  stEl.textContent=st;
  stEl.style.color=st==='CRITICAL'?'#ef4444':st==='WARNING'?'#f59e0b':'#10b981';

  // State details
  document.getElementById('state-details').innerHTML=[
    ['Task',     obs.task_id||'—'],
    ['Attitude', obs.attitude_error_deg!=null?obs.attitude_error_deg.toFixed(1)+'°':'—'],
    ['Bus V',    obs.bus_voltage!=null?obs.bus_voltage.toFixed(2)+'V':'—'],
    ['BER',      obs.bit_error_rate!=null?obs.bit_error_rate.toFixed(4):'—'],
    ['Obs Window',obs.observation_window_active?'✅ Active (+0.12)':'—'],
  ].map(([k,v])=>`<div style="display:flex;justify-content:space-between"><span style="color:var(--muted)">${k}</span><span>${v}</span></div>`).join('');
}

function updatePhase(meta){
  // Safe null check — this was the bug
  if(!meta||meta===null)return;
  const phase=meta.phase||0;
  const step=meta.phase_step||0;
  const names=['EPS Crisis (steps 1-12)','Thermal Crisis (steps 13-24)','Comms Crisis (steps 25-36)'];
  for(let i=0;i<3;i++){
    const el=document.getElementById('ph'+i);
    if(i<phase) el.className='phase-seg done';
    else if(i===phase) el.className='phase-seg active';
    else el.className='phase-seg';
  }
  document.getElementById('phase-label').textContent=
    `Phase ${phase+1}: ${names[Math.min(phase,2)]}  (step ${step+1}/12)`;
}

function updateChart(reward){
  rewardHistory.push(reward);
  if(rewardHistory.length>36)rewardHistory.shift();
  const n=rewardHistory.length,W=260,H=80;
  const pts=rewardHistory.map((r,i)=>{
    const x=n===1?W/2:(i/(n-1))*W;
    const y=H-(Math.max(0,Math.min(1,r))*(H-8)+4);
    return x.toFixed(1)+','+y.toFixed(1);
  }).join(' ');
  document.getElementById('reward-line').setAttribute('points',pts);
  const avg=rewardHistory.reduce((a,b)=>a+b,0)/rewardHistory.length;
  document.getElementById('avg-reward').textContent=avg.toFixed(4);
  document.getElementById('episode-score').textContent='Score: '+avg.toFixed(3);
}

function updateAgent(data){
  const ad=data.agent_decision;
  if(!ad)return;
  document.getElementById('agent-decision-box').innerHTML=
    `<div class="decision-action">→ ${ad.recommended_action}</div>
     <div class="decision-rationale">${ad.rationale}</div>`;
  if(ad.specialists){
    document.getElementById('specialist-list').innerHTML=
      Object.entries(ad.specialists).map(([name,s])=>
        `<div class="specialist-row">
          <span class="spec-name">${name.replace('_Specialist','')}</span>
          <span class="spec-action">${s.action}</span>
          <span class="spec-conf">${Math.round(s.confidence*100)}%</span>
        </div>`).join('');
  }
}

function addLog(line,cls=''){
  const feed=document.getElementById('log-feed');
  const div=document.createElement('div');
  div.innerHTML=`<span class="${cls}">${line}</span>`;
  feed.appendChild(div);
  feed.scrollTop=feed.scrollHeight;
}

async function startTask(task,btn){
  document.querySelectorAll('.task-btn').forEach(b=>b.classList.remove('active'));
  if(btn)btn.classList.add('active');
  rewardHistory=[];stepCount=0;
  document.getElementById('step-num').textContent='0';
  document.getElementById('log-feed').innerHTML='';
  addLog('Starting '+task+' task...','log-action');
  try{
    const res=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:task})});
    const data=await res.json();
    const obs=data.observation;
    updateTelemetry(obs);
    // Safe: use world_model from response, not obs.metadata
    const wm=data.world_model||{};
    updateFaults(wm.fault_beliefs||{});
    updatePhase(wm);  // pass world_model which always has phase/phase_step
    document.getElementById('dominant-sys').textContent=wm.dominant_subsystem||'—';
    addLog(`Reset: soc=${obs.battery_soc?.toFixed(1)}% temp=${obs.thermal_temp?.toFixed(1)}C sunlit=${obs.sunlit}`,'log-reward');
  }catch(e){addLog('Error: '+e.message,'log-warn')}
}

async function takeAction(action){
  stepCount++;
  document.getElementById('step-num').textContent=stepCount;
  try{
    const res=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action_type:action})});
    const data=await res.json();
    const obs=data.observation;
    updateTelemetry(obs);
    updateChart(data.reward||0);
    updateAgent(data);
    // Safe: use world_model from response, not obs.metadata
    const wm=data.world_model||{};
    updateFaults(wm.fault_beliefs||{});
    updatePhase(wm);  // world_model always has phase/phase_step
    document.getElementById('dominant-sys').textContent=wm.dominant_subsystem||'—';
    const cls=data.reward>0.6?'log-reward':data.reward>0.4?'':'log-warn';
    addLog(`[${stepCount}] ${action} → reward=${(data.reward||0).toFixed(4)}${data.done?' DONE':''}`,cls);
    if(data.done)addLog('Episode complete.','log-action');
  }catch(e){addLog('Error: '+e.message,'log-warn')}
}

initFaultList();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD_HTML


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()