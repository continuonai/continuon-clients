from pathlib import Path


TRAINING_PLAN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Training Plan</title>
    <style>
        body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: #0d0d0d; color: #e0e0e0; margin: 0; padding: 32px; }
        h1 { margin-top: 0; }
        a { color: #00ff88; }
        .card { background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        .muted { color: #888; font-size: 0.9em; }
        code { background: #111; padding: 2px 4px; border-radius: 4px; }
        pre { background: #111; padding: 12px; border-radius: 8px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Continuon Studio · Training Plan</h1>
    <p class="muted">Pi5 seed → RLDS capture → cloud train → Hope Model v1 OTA.</p>

    <div class="card">
        <h3>Quick Links</h3>
        <ul>
            <li><a href="/docs/training-plan.md" target="_blank">Full training plan (docs)</a></li>
            <li><code>python -m continuonbrain.services.training_manager --help</code></li>
        </ul>
    </div>

    <div class="card">
        <h3>On-device steps</h3>
        <ol>
            <li>Health: <code>PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick</code></li>
            <li>Collect RLDS (≥16 episodes) under <code>/opt/continuonos/brain/rlds/episodes</code></li>
            <li>Local train: <code>python -m continuonbrain.trainer.local_lora_trainer --config continuonbrain/configs/pi5-donkey.json --use-stub-hooks</code></li>
            <li>Export RLDS: <code>python -m continuonbrain.rlds.export_pipeline --episodes-dir /opt/continuonos/brain/rlds/episodes --output-dir /opt/continuonos/brain/rlds/export</code></li>
        </ol>
    </div>

    <div class="card">
        <h3>Cloud + OTA</h3>
        <ol>
            <li>Upload RLDS/export to cloud bucket.</li>
            <li>Train on TPU: <code>python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000</code></li>
            <li>Assemble signed edge bundle (see <code>docs/bundle_manifest.md</code>).</li>
            <li>OTA back to Pi via companion app; verify and hot-swap.</li>
        </ol>
    </div>

    <div class="card">
        <h3>Post-deploy validation</h3>
        <p><code>PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware</code></p>
        <p class="muted">Check RLDS sync, control loop latency, and keep rollback ready.</p>
    </div>
</body>
</html>
"""


def get_training_plan_html() -> str:
    return TRAINING_PLAN_HTML

