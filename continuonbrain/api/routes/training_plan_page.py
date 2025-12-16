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
    <p class="muted">JAX-first seed: Pi RLDS → TFRecord → TPU → OTA bundle + proof.</p>

    <div class="card">
        <h3>Quick Links</h3>
        <ul>
            <li><a href="/docs/training-plan.md" target="_blank">Full training plan (docs)</a></li>
            <li><a href="/docs/jax-training-pipeline.md" target="_blank">JAX training pipeline</a></li>
            <li><code>python -m continuonbrain.services.training_manager --help</code></li>
            <li><code>python -m continuonbrain.jax_models.train.local_sanity_check --help</code></li>
        </ul>
    </div>

    <div class="card">
        <h3>On-device capture & proof</h3>
        <ol>
            <li>Health: <code>PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick</code></li>
            <li>Collect RLDS (≥16 episodes) under <code>/opt/continuonos/brain/rlds/episodes</code> (JSON/JSONL or TFRecord).</li>
            <li>JAX sanity check (uses TFRecord or JSON fallback):<br>
                <code>python -m continuonbrain.jax_models.train.local_sanity_check --rlds-dir /opt/continuonos/brain/rlds/episodes --arch-preset pi5 --max-steps 8 --batch-size 4 --metrics-path /tmp/jax_sanity.csv --checkpoint-dir /tmp/jax_ckpts</code>
            </li>
            <li>Imagination proof metrics: the sanity check now logs <code>mse_main</code> and <code>mse_imagination_tail</code> (planner/tool supervision packed into the tail of the action vector).</li>
            <li>Studio proof page: <a href="/training_proof"><code>/training_proof</code></a> plots <code>mse_main</code> vs <code>mse_imagination_tail</code> using <code>/api/training/status</code>.</li>
            <li>Convert RLDS to TFRecord for TPU: <code>python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir /opt/continuonos/brain/rlds/episodes --output-dir /opt/continuonos/brain/rlds/tfrecord --compress</code></li>
            <li>Proof-of-learning demo (background learner): <code>python prove_learning_capability.py</code> → saves <code>proof_of_learning.json</code> for audit.</li>
        </ol>
    </div>

    <div class="card">
        <h3>Cloud train + bundle</h3>
        <ol>
            <li>Upload <code>/opt/continuonos/brain/rlds/tfrecord</code> to your bucket.</li>
            <li>TPU train (JAX): <code>python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000</code></li>
            <li>Export: <code>python -m continuonbrain.jax_models.export.export_jax --checkpoint-path gs://... --output-path ./models/core_model_inference --quantization fp16</code><br>
                Optional Hailo: <code>python -m continuonbrain.jax_models.export.export_hailo --checkpoint-path ./models/core_model_inference --output-dir ./models/core_model_hailo</code></li>
            <li>Assemble signed edge bundle + <code>edge_manifest.json</code> (see <code>docs/bundle_manifest.md</code>), include latest proof + metrics.</li>
            <li>OTA back to Pi via companion app; verify and hot-swap.</li>
        </ol>
    </div>

    <div class="card">
        <h3>Post-deploy validation</h3>
        <p><code>PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware</code></p>
        <p><code>python -m continuonbrain.jax_models.export.infer_cpu --model-path ./models/core_model_inference --obs-dim 128 --action-dim 32</code></p>
        <p class="muted">Check RLDS sync, control loop latency, and keep rollback ready. Stash <code>proof_of_learning.json</code> beside the deployed bundle for review.</p>
    </div>
</body>
</html>
"""


def get_training_plan_html() -> str:
    return TRAINING_PLAN_HTML

