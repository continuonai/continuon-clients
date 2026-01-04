/**
 * continuon.cloud Cloud Functions
 *
 * RLDS Episode Ingestion and Remote Robot Connection APIs
 */

import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import { Storage } from "@google-cloud/storage";
import { v4 as uuidv4 } from "uuid";

// Initialize Firebase Admin
admin.initializeApp();
const db = admin.firestore();
const storage = new Storage();

// Configuration
const RLDS_BUCKET = "continuon-rlds";
const SIGNED_URL_EXPIRY_MINUTES = 60;

// ============================================================================
// RLDS Episode Ingestion APIs
// ============================================================================

/**
 * Generate a signed upload URL for RLDS episode upload
 *
 * POST /api/rlds/signed-url
 * {
 *   "robot_id": "14d4b680",
 *   "episode_id": "ep_1704384000000",
 *   "metadata": { "duration_ms": 30000, "step_count": 300 }
 * }
 */
export const rldsSignedUrl = functions.https.onRequest(async (req, res) => {
  // CORS
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { robot_id, episode_id, metadata } = req.body;

    if (!robot_id || !episode_id) {
      res.status(400).json({ error: "robot_id and episode_id required" });
      return;
    }

    // Verify robot is registered
    const robotDoc = await db.collection("rcan_registry").doc(robot_id).get();
    if (!robotDoc.exists) {
      res.status(404).json({ error: "Robot not registered" });
      return;
    }

    // Generate file path
    const episodePath = `episodes/raw/${robot_id}/${episode_id}/episode.tar.gz`;
    const bucket = storage.bucket(RLDS_BUCKET);
    const file = bucket.file(episodePath);

    // Generate signed URL for upload
    const [signedUrl] = await file.getSignedUrl({
      version: "v4",
      action: "write",
      expires: Date.now() + SIGNED_URL_EXPIRY_MINUTES * 60 * 1000,
      contentType: "application/gzip",
    });

    // Create episode record in Firestore
    const episodeRef = db.collection("rlds_episodes").doc(episode_id);
    await episodeRef.set({
      robot_id,
      episode_id,
      status: "pending_upload",
      storage_ref: `gs://${RLDS_BUCKET}/${episodePath}`,
      metadata: metadata || {},
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
    });

    res.json({
      success: true,
      upload_url: signedUrl,
      expires_at: new Date(Date.now() + SIGNED_URL_EXPIRY_MINUTES * 60 * 1000).toISOString(),
      episode_ref: episodePath,
      episode_id,
    });
  } catch (error) {
    functions.logger.error("Error generating signed URL:", error);
    res.status(500).json({ error: "Failed to generate upload URL" });
  }
});

/**
 * Validate and ingest an uploaded RLDS episode
 *
 * POST /api/rlds/ingest
 * {
 *   "episode_id": "ep_1704384000000",
 *   "checksum": "sha256:abc123..."
 * }
 */
export const rldsIngest = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { episode_id, checksum } = req.body;

    if (!episode_id) {
      res.status(400).json({ error: "episode_id required" });
      return;
    }

    // Get episode record
    const episodeRef = db.collection("rlds_episodes").doc(episode_id);
    const episodeDoc = await episodeRef.get();

    if (!episodeDoc.exists) {
      res.status(404).json({ error: "Episode not found" });
      return;
    }

    const episodeData = episodeDoc.data()!;

    // Verify file exists in storage
    const bucket = storage.bucket(RLDS_BUCKET);
    const filePath = episodeData.storage_ref.replace(`gs://${RLDS_BUCKET}/`, "");
    const file = bucket.file(filePath);
    const [exists] = await file.exists();

    if (!exists) {
      res.status(400).json({ error: "Episode file not uploaded" });
      return;
    }

    // TODO: Verify checksum
    // const [metadata] = await file.getMetadata();
    // if (checksum && metadata.md5Hash !== checksum) {
    //   res.status(400).json({ error: "Checksum mismatch" });
    //   return;
    // }

    // Move to validated folder
    const validatedPath = filePath.replace("episodes/raw/", "episodes/validated/");
    await file.copy(bucket.file(validatedPath));

    // Update episode status
    await episodeRef.update({
      status: "validated",
      validated_storage_ref: `gs://${RLDS_BUCKET}/${validatedPath}`,
      checksum: checksum || null,
      validated_at: admin.firestore.FieldValue.serverTimestamp(),
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
      training_eligible: true,
    });

    res.json({
      success: true,
      status: "validated",
      episode_id,
      training_eligible: true,
    });
  } catch (error) {
    functions.logger.error("Error ingesting episode:", error);
    res.status(500).json({ error: "Failed to ingest episode" });
  }
});

/**
 * List episodes for a robot (for Colab training)
 *
 * GET /api/rlds/list?robot_id=14d4b680&status=validated
 */
export const rldsList = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { robot_id, status, limit } = req.query;

    let query = db.collection("rlds_episodes") as FirebaseFirestore.Query;

    if (robot_id) {
      query = query.where("robot_id", "==", robot_id);
    }
    if (status) {
      query = query.where("status", "==", status);
    }

    query = query.orderBy("created_at", "desc").limit(Number(limit) || 100);

    const snapshot = await query.get();
    const episodes = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data(),
      created_at: doc.data().created_at?.toDate?.()?.toISOString(),
      validated_at: doc.data().validated_at?.toDate?.()?.toISOString(),
    }));

    res.json({
      success: true,
      count: episodes.length,
      episodes,
    });
  } catch (error) {
    functions.logger.error("Error listing episodes:", error);
    res.status(500).json({ error: "Failed to list episodes" });
  }
});

// ============================================================================
// Remote Connection APIs
// ============================================================================

/**
 * Verify robot ownership for remote connection
 *
 * POST /api/remote/verify
 * {
 *   "robot_ruri": "rcan://continuon.cloud/continuon/companion-v1/14d4b680",
 *   "user_uid": "firebase-uid",
 *   "local_claim_proof": "signed-token"
 * }
 */
export const remoteVerify = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { robot_ruri, user_uid, local_claim_proof } = req.body;

    if (!robot_ruri || !user_uid) {
      res.status(400).json({ error: "robot_ruri and user_uid required" });
      return;
    }

    // Extract device_id from RURI
    const ruriParts = robot_ruri.split("/");
    const device_id = ruriParts[ruriParts.length - 1];

    // Verify robot exists and check ownership
    const robotDoc = await db.collection("rcan_registry").doc(device_id).get();
    if (!robotDoc.exists) {
      res.status(404).json({ error: "Robot not found", verified: false });
      return;
    }

    const robotData = robotDoc.data()!;

    // Check if user is the owner
    // TODO: Implement proper ownership verification with local_claim_proof
    const isOwner = robotData.owner_uid === user_uid ||
                    robotData.owner_email === user_uid;

    if (!isOwner && !local_claim_proof) {
      res.status(403).json({
        error: "Not authorized. Local claim required first.",
        verified: false
      });
      return;
    }

    // Create remote session
    const sessionId = uuidv4();
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000); // 1 hour

    await db.collection("remote_sessions").doc(sessionId).set({
      robot_ruri,
      device_id,
      user_uid,
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      expires_at: admin.firestore.Timestamp.fromDate(expiresAt),
      status: "pending",
      local_claim_verified: !!local_claim_proof,
      connection_type: "webrtc",
    });

    res.json({
      success: true,
      verified: true,
      session_id: sessionId,
      session_token: sessionId, // In production, use a signed JWT
      expires_at: expiresAt.toISOString(),
      robot_endpoint: robotData.endpoint,
      relay_endpoint: `wss://relay.continuon.cloud/${device_id}`,
    });
  } catch (error) {
    functions.logger.error("Error verifying remote access:", error);
    res.status(500).json({ error: "Verification failed", verified: false });
  }
});

/**
 * Get remote session status
 *
 * GET /api/remote/session?session_id=xyz
 */
export const remoteSession = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");

  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { session_id } = req.query;

    if (!session_id) {
      res.status(400).json({ error: "session_id required" });
      return;
    }

    const sessionDoc = await db.collection("remote_sessions").doc(session_id as string).get();

    if (!sessionDoc.exists) {
      res.status(404).json({ error: "Session not found" });
      return;
    }

    const sessionData = sessionDoc.data()!;
    const isExpired = sessionData.expires_at.toDate() < new Date();

    res.json({
      success: true,
      session_id,
      status: isExpired ? "expired" : sessionData.status,
      robot_ruri: sessionData.robot_ruri,
      expires_at: sessionData.expires_at.toDate().toISOString(),
      connection_type: sessionData.connection_type,
    });
  } catch (error) {
    functions.logger.error("Error getting session:", error);
    res.status(500).json({ error: "Failed to get session" });
  }
});

// ============================================================================
// Training Job APIs (for Colab integration)
// ============================================================================

/**
 * Create a training job
 *
 * POST /api/training/create
 * {
 *   "type": "seed" | "slow_loop",
 *   "episode_ids": ["ep_1", "ep_2"],
 *   "config": { ... }
 * }
 */
export const trainingCreate = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { type, episode_ids, config, robot_model } = req.body;

    if (!type || !episode_ids || episode_ids.length === 0) {
      res.status(400).json({ error: "type and episode_ids required" });
      return;
    }

    const jobId = `job_${Date.now()}_${uuidv4().substring(0, 8)}`;

    await db.collection("training_jobs").doc(jobId).set({
      job_id: jobId,
      type,
      status: "queued",
      robot_model: robot_model || "companion-v1",
      episode_ids,
      config: config || {},
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      started_at: null,
      completed_at: null,
      output: null,
      colab_notebook_url: null,
    });

    // Generate Colab notebook URL (template with pre-filled params)
    const colabParams = new URLSearchParams({
      job_id: jobId,
      type,
      episode_count: episode_ids.length.toString(),
    });
    const colabUrl = `https://colab.research.google.com/github/anthropics/continuonxr/blob/main/notebooks/${type}_training.ipynb?${colabParams}`;

    res.json({
      success: true,
      job_id: jobId,
      status: "queued",
      episode_count: episode_ids.length,
      colab_notebook_url: colabUrl,
    });
  } catch (error) {
    functions.logger.error("Error creating training job:", error);
    res.status(500).json({ error: "Failed to create training job" });
  }
});

/**
 * Update training job status (called from Colab)
 *
 * POST /api/training/update
 */
export const trainingUpdate = functions.https.onRequest(async (req, res) => {
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const { job_id, status, output, model_ref, metrics } = req.body;

    if (!job_id || !status) {
      res.status(400).json({ error: "job_id and status required" });
      return;
    }

    const updateData: Record<string, unknown> = {
      status,
      updated_at: admin.firestore.FieldValue.serverTimestamp(),
    };

    if (status === "running") {
      updateData.started_at = admin.firestore.FieldValue.serverTimestamp();
    } else if (status === "completed" || status === "failed") {
      updateData.completed_at = admin.firestore.FieldValue.serverTimestamp();
      if (output) updateData.output = output;
      if (model_ref) updateData["output.model_ref"] = model_ref;
      if (metrics) updateData["output.metrics"] = metrics;
    }

    await db.collection("training_jobs").doc(job_id).update(updateData);

    res.json({ success: true, job_id, status });
  } catch (error) {
    functions.logger.error("Error updating training job:", error);
    res.status(500).json({ error: "Failed to update training job" });
  }
});

// Export all functions
export const rlds = functions.https.onRequest(async (req, res) => {
  const path = req.path.replace("/api/rlds", "");

  if (path === "/signed-url" || path === "/signed-url/") {
    return rldsSignedUrl(req, res);
  } else if (path === "/ingest" || path === "/ingest/") {
    return rldsIngest(req, res);
  } else if (path === "/list" || path === "/list/") {
    return rldsList(req, res);
  } else {
    res.status(404).json({ error: "Endpoint not found" });
  }
});

export const remote = functions.https.onRequest(async (req, res) => {
  const path = req.path.replace("/api/remote", "");

  if (path === "/verify" || path === "/verify/") {
    return remoteVerify(req, res);
  } else if (path === "/session" || path === "/session/") {
    return remoteSession(req, res);
  } else {
    res.status(404).json({ error: "Endpoint not found" });
  }
});

export const training = functions.https.onRequest(async (req, res) => {
  const path = req.path.replace("/api/training", "");

  if (path === "/create" || path === "/create/") {
    return trainingCreate(req, res);
  } else if (path === "/update" || path === "/update/") {
    return trainingUpdate(req, res);
  } else {
    res.status(404).json({ error: "Endpoint not found" });
  }
});
