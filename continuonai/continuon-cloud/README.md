# Continuon-Cloud (staging docs)

This folder remains a staging/spec area for Continuon-Cloud. Production ingestion/training code lives in the dedicated cloud repo. Keep this aligned with `docs/monorepo-structure.md` and root README; no secrets or production scripts here.

## Google Cloud environment (staging) — reference setup

- **Project**: `continuon-rlds-project` (example)
- **Region**: pick a single region for TPU + buckets (e.g., `us-central1`)
- **Buckets**:
  - `gs://continuon-rlds` — RLDS TFRecords (`rlds/episodes/`), checkpoints (`checkpoints/core_model_v0/`), and export artifacts.
  - (Optional) `gs://continuon-rlds-staging` — signed ingest drop.
- **APIs to enable**: `compute.googleapis.com`, `tpu.googleapis.com`, `storage.googleapis.com`, `artifactregistry.googleapis.com` (if using custom images).
- **Service accounts**:
  - Trainer/TPU: `tpu-trainer@continuon-rlds-project.iam.gserviceaccount.com`
  - Ingest/uploader (if used): `ingest-uploader@continuon-rlds-project.iam.gserviceaccount.com`
- **IAM (minimum)**:
  - `roles/storage.objectAdmin` on `continuon-rlds` for the trainer SA.
  - `roles/storage.objectCreator` on staging bucket for uploader SA; no delete in staging by default.
  - `roles/tpu.admin` (or scoped custom) for TPU job launcher account.
- **Images/Deps**:
  - TPU jobs: JAX TPU wheels, `flax`, `optax`, `tensorflow` (for TFRecord ingest), `orbax-checkpoint`, `google-cloud-storage`.
  - If using custom container: push to Artifact Registry in the same region.

### One-time gcloud setup (example)

```bash
gcloud projects create continuon-rlds-project
gcloud services enable compute.googleapis.com tpu.googleapis.com storage.googleapis.com artifactregistry.googleapis.com --project continuon-rlds-project

gsutil mb -c STANDARD -l us-central1 gs://continuon-rlds
# Optional staging bucket
gsutil mb -c STANDARD -l us-central1 gs://continuon-rlds-staging

gcloud iam service-accounts create tpu-trainer --project continuon-rlds-project
gcloud iam service-accounts create ingest-uploader --project continuon-rlds-project

gcloud storage buckets add-iam-policy-binding gs://continuon-rlds \
  --member=serviceAccount:tpu-trainer@continuon-rlds-project.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

gcloud storage buckets add-iam-policy-binding gs://continuon-rlds-staging \
  --member=serviceAccount:ingest-uploader@continuon-rlds-project.iam.gserviceaccount.com \
  --role=roles/storage.objectCreator
```

### IAM checklist
- TPU trainer SA (`tpu-trainer@…`):
  - `roles/storage.objectAdmin` on `gs://continuon-rlds`
  - `roles/tpu.admin` (or scoped TPU runner) on the project
  - Optional: `roles/artifactregistry.reader` if using custom containers
- Uploader SA (`ingest-uploader@…`):
  - `roles/storage.objectCreator` on staging bucket
  - No delete on staging by default
- Your CLI user/group: `roles/owner` or least-privileged set to create buckets/SAs/TPU VMs

## TPU job submission (example, v5e)

Assumes: project `continuon-rlds-project`, zone `us-central1-b`, bucket `gs://continuon-rlds`, service account `tpu-trainer@continuon-rlds-project.iam.gserviceaccount.com`, and you have a startup script that installs deps and runs the trainer.

```bash
export PROJECT_ID=continuon-rlds-project
export ZONE=us-central1-b
export BUCKET=gs://continuon-rlds

gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

gsutil cp continuonai/continuon-cloud/startup-script.sh $BUCKET/

gcloud alpha tpu v5e create continuon-seed-tpu \
  --project $PROJECT_ID \
  --zone $ZONE \
  --accelerator-type v5e-1 \
  --version tpu-ubuntu2204-base \
  --metadata=startup-script-url=$BUCKET/startup-script.sh \
  --service-account tpu-trainer@${PROJECT_ID}.iam.gserviceaccount.com
```

Delete when done:
```bash
gcloud alpha tpu v5e delete continuon-seed-tpu --project $PROJECT_ID --zone $ZONE
```

Reminder: clean up `startup-script.sh` in the bucket or rotate it after use to avoid stale scripts lingering in OTA/GCS paths.

## Custom container build/push (Artifact Registry, optional)

If you prefer a containerized TPU job, build and push an image with deps baked in.

```bash
export PROJECT_ID=continuon-rlds-project
export REGION=us-central1
export REPO=continuon-tpu
export IMAGE=seed-trainer

gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION --project=$PROJECT_ID

cat > Dockerfile <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    flax optax tensorflow orbax-checkpoint google-cloud-storage
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "continuonbrain.jax_models.train.cloud.tpu_train"]
EOF

docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest .
gcloud auth configure-docker $REGION-docker.pkg.dev --project $PROJECT_ID
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest
```

To run with the container (conceptual TPU VM launch):
```bash
gcloud alpha tpu v5e create continuon-seed-tpu \
  --project $PROJECT_ID \
  --zone us-central1-b \
  --accelerator-type v5e-1 \
  --version tpu-ubuntu2204-base \
  --metadata=startup-script='#!/bin/bash
    sudo apt-get update && sudo apt-get install -y docker.io
    systemctl start docker
    docker run --rm \
      -e PYTHONPATH=/app \
      $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest \
      --data-path gs://continuon-rlds/rlds/episodes \
      --output-dir gs://continuon-rlds/checkpoints/core_model_v0 \
      --batch-size 256 --num-steps 10000 --learning-rate 1e-4
  ' \
  --service-account tpu-trainer@${PROJECT_ID}.iam.gserviceaccount.com
```

## Staging ingest expectations

- All uploads into the staging bucket must be signed client-side and include a manifest with provenance metadata (Continuon Brain runtime + Continuon AI app versions), environment ID, deterministic package ID, and SHA-256 checksums for the archive and every episode blob.
- The ingestion service rejects unsigned, unverifiable, or checksum-mismatched artifacts before they ever reach the staging bucket and surfaces 4xx errors to clients; keep local copies until a signed re-upload succeeds.
- See [signed-ingestion.md](./signed-ingestion.md) for required manifest fields and verification steps before promotion to training storage.
