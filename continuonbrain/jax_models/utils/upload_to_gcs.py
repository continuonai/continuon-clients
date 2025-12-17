"""
RLDS Upload Pipeline to GCS

Upload TFRecord episodes to GCS bucket with provenance and metadata.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
    
    Returns:
        Hex digest of file hash
    """
    sha256 = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def upload_episode_to_gcs(
    episode_path: Path,
    gcs_bucket: str,
    gcs_prefix: str = "rlds/episodes",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Upload a single TFRecord episode to GCS.
    
    Args:
        episode_path: Path to TFRecord episode file
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        metadata: Optional metadata dictionary
    
    Returns:
        Dictionary with upload information (gcs_path, hash, etc.)
    """
    if not GCS_AVAILABLE:
        raise ImportError("Google Cloud Storage is required. Install with: pip install google-cloud-storage")
    
    # Compute file hash
    file_hash = compute_file_hash(episode_path)
    
    # Determine GCS path
    gcs_filename = f"{episode_path.stem}_{file_hash[:8]}.tfrecord"
    if episode_path.suffix == '.gz':
        gcs_filename += '.gz'
    gcs_path = f"{gcs_prefix}/{gcs_filename}"
    
    # Upload file
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)
    
    print(f"Uploading {episode_path.name} to gs://{gcs_bucket}/{gcs_path}...")
    blob.upload_from_filename(str(episode_path))
    
    # Upload metadata if provided
    metadata_path = None
    if metadata:
        metadata_json = json.dumps(metadata, indent=2, default=str)
        metadata_filename = f"{episode_path.stem}_metadata.json"
        metadata_gcs_path = f"{gcs_prefix}/metadata/{metadata_filename}"
        metadata_blob = bucket.blob(metadata_gcs_path)
        metadata_blob.upload_from_string(metadata_json, content_type='application/json')
        metadata_path = metadata_gcs_path
    
    return {
        'episode_path': str(episode_path),
        'gcs_path': f"gs://{gcs_bucket}/{gcs_path}",
        'metadata_path': f"gs://{gcs_bucket}/{metadata_path}" if metadata_path else None,
        'hash': file_hash,
        'size_bytes': episode_path.stat().st_size,
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
    }


def upload_directory_to_gcs(
    local_dir: Path,
    gcs_bucket: str,
    gcs_prefix: str = "rlds/episodes",
    pattern: str = "*.tfrecord*",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Upload all TFRecord episodes in a directory to GCS.
    
    Args:
        local_dir: Local directory containing episodes
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix/path
        pattern: Glob pattern for episode files
        metadata: Optional metadata dictionary (applied to all episodes)
    
    Returns:
        List of upload information dictionaries
    """
    local_dir = Path(local_dir)
    
    # Find episode files
    tfrecord_files = list(local_dir.glob("*.tfrecord"))
    tfrecord_gz_files = list(local_dir.glob("*.tfrecord.gz"))
    episode_files = tfrecord_files + tfrecord_gz_files
    
    if not episode_files:
        print(f"No TFRecord files found in {local_dir}")
        return []
    
    print(f"Found {len(episode_files)} episode files")
    
    upload_results = []
    for episode_file in episode_files:
        try:
            result = upload_episode_to_gcs(
                episode_file,
                gcs_bucket,
                gcs_prefix,
                metadata=metadata,
            )
            upload_results.append(result)
        except Exception as e:
            print(f"Error uploading {episode_file}: {e}")
            continue
    
    # Create upload manifest
    manifest = {
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
        'source_directory': str(local_dir),
        'gcs_bucket': gcs_bucket,
        'gcs_prefix': gcs_prefix,
        'episodes': upload_results,
        'total_episodes': len(upload_results),
    }
    
    # Upload manifest
    if GCS_AVAILABLE:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        manifest_path = f"{gcs_prefix}/manifest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        manifest_blob = bucket.blob(manifest_path)
        manifest_blob.upload_from_string(
            json.dumps(manifest, indent=2, default=str),
            content_type='application/json'
        )
        print(f"\n✅ Upload manifest: gs://{gcs_bucket}/{manifest_path}")
    
    print(f"\n✅ Uploaded {len(upload_results)}/{len(episode_files)} episodes")
    
    return upload_results


def main():
    """CLI entry point for GCS upload."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload RLDS episodes to GCS")
    parser.add_argument("--local-dir", type=Path, required=True, help="Local directory with TFRecord episodes")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--gcs-prefix", type=str, default="rlds/episodes", help="GCS prefix/path")
    parser.add_argument("--metadata", type=Path, help="Path to metadata JSON file")
    
    args = parser.parse_args()
    
    metadata = None
    if args.metadata:
        with args.metadata.open('r') as f:
            metadata = json.load(f)
    
    upload_directory_to_gcs(
        args.local_dir,
        args.gcs_bucket,
        args.gcs_prefix,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()

