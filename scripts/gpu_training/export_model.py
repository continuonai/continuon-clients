#!/usr/bin/env python3
"""
Export trained model for deployment to Raspberry Pi.
Creates a portable checkpoint that can be loaded on the Pi.
"""

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def create_export_manifest(checkpoint_dir: Path, output_path: Path) -> dict:
    """Create manifest for the exported model."""
    manifest = {
        "format": "continuonbrain_export",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "source": "gpu_training",
        "compatible_with": ["pi5", "jetson", "cpu"],
        "files": [],
        "config": {},
    }
    
    # Find checkpoint files
    for f in checkpoint_dir.glob("*"):
        if f.is_file():
            manifest["files"].append({
                "name": f.name,
                "size": f.stat().st_size,
            })
    
    # Load training config if exists
    config_file = checkpoint_dir / "model_final.json"
    if config_file.exists():
        with open(config_file) as f:
            manifest["config"] = json.load(f)
    
    return manifest


def export_model(checkpoint_dir: Path, output_path: Path):
    """Export model checkpoint as a zip file."""
    print("=" * 60)
    print("  ContinuonBrain Model Export")
    print("=" * 60)
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    print(f"Source: {checkpoint_dir}")
    print(f"Output: {output_path}")
    
    # Create manifest
    manifest = create_export_manifest(checkpoint_dir, output_path)
    
    # Write manifest to temp location
    manifest_path = checkpoint_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Create zip file
    print("\nCreating export package...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in checkpoint_dir.glob("*"):
            if f.is_file():
                arcname = f.name
                zf.write(f, arcname)
                print(f"  Added: {arcname}")
    
    # Cleanup temp manifest
    manifest_path.unlink()
    
    # Stats
    export_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExport complete!")
    print(f"  Size: {export_size:.1f} MB")
    print(f"  File: {output_path}")
    
    print("\n" + "=" * 60)
    print("  Deployment Instructions")
    print("=" * 60)
    print("""
To deploy to Raspberry Pi:

1. Copy the export file to Pi:
   scp {output} pi@<pi-ip>:/tmp/

2. On the Pi, install the model:
   curl -X POST http://localhost:8081/api/training/install_bundle \\
     -F "file=@/tmp/{filename}" \\
     -F "kind=jax_seed_manifest"

Or manually:
   unzip /tmp/{filename} -d /opt/continuonos/brain/model/adapters/candidate/
   
3. Activate the model:
   curl -X POST http://localhost:8081/api/admin/promote_candidate
""".format(output=output_path, filename=output_path.name))
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Export trained model for Pi deployment")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                        help="Directory containing training checkpoints")
    parser.add_argument("--output", type=Path, 
                        default=Path(f"model_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"),
                        help="Output zip file path")
    
    args = parser.parse_args()
    
    success = export_model(args.checkpoint_dir, args.output)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

