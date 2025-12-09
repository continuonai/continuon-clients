# Agent Instructions (ContinuonBrain)

Scope: `continuonbrain/`.

- The Continuon Brain runtime and scaffolding are now co-located in this monorepo. Keep docs clear about what is production-ready versus staged scaffolding so downstream consumers can promote pieces confidently.
- Prefer small, dependency-light utilities. Avoid adding heavy ML packages beyond what the trainer stubs already expect; keep optional imports guarded so Pi/Jetson environments can still import modules.
- Maintain alignment with RLDS inputs/outputs: update comments/examples when changing schema expectations, and keep sample manifests/configs consistent with the trainer defaults.
- Use type hints and clear docstrings for trainer hooks/safety adapters. Keep safety gating logic explicit and easy to override from continuonos.
- Testing expectations:
  - For Python modules, run `python -m continuonbrain.trainer.local_lora_trainer --help` or an equivalent import check after altering trainer code.
  - For manifest/config changes, validate JSON syntax and ensure sample paths remain coherent with README references.
  Mention any skipped checks due to unavailable dependencies or hardware.

## Production Installation System

**Context**: We're packaging ContinuonBrain as a single `.deb` package with kiosk mode for production deployment.

**Guidelines for packaging work:**
- Keep package structure in `packaging/continuonbrain/` directory
- Follow Debian packaging standards (control files, scripts, etc.)
- Test installation scripts on clean systems before committing
- Document any new dependencies in `DEBIAN/control`
- Maintain backward compatibility with manual installation during transition
- Use systemd for service management (no custom init scripts)
- Follow security best practices (least privilege, sandboxing)
- Keep CLI tools simple and user-friendly

**Key Files:**
- `packaging/continuonbrain/DEBIAN/*` - Package control files
- `packaging/continuonbrain/etc/systemd/system/*` - Service definitions
- `packaging/continuonbrain/usr/bin/*` - CLI tools
- `continuonbrain/watchdog.py` - Health monitoring service
- `build-package.sh` - Package build script

**Testing:**
- Build: `./build-package.sh`
- Install: `sudo apt install ./build/continuonbrain_1.0.0_arm64.deb`
- Verify: `continuonbrain status`
- Uninstall: `sudo apt remove continuonbrain`

**Phases:**
1. ✅ Debian Package Structure (complete)
2. 🔄 Kiosk Mode (next)
3. 📋 Developer Mode
4. 📋 Installation Image
5. 📋 Error Recovery

See `packaging/implementation_plan.md` for detailed roadmap.
