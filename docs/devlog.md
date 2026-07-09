# picoforge devlog

Two lines per session: what was done, what comes next. Newest entry first.

## 2026-07-09 — Environment setup complete

Toolchain verified (Xcode + Metal Toolchain 17F109, macOS 26.5.1, Apple clang 17).
Metal 4 probe compiled and run on the M5 Pro: Apple10 + Metal4 families supported,
MTLTensor creation works — Phase 2 is viable on this machine (probe:
`tools/probe/metal4_probe.m`). Working set reported: 55.7 GB. Python venv under
`tools/venv` (numpy 2.5.1, torch 2.13.0, transformers 5.13.0), versions frozen in
`tools/requirements.txt`. Qwen3-0.6B downloaded to `models/` (gitignored), SHA256s
in `docs/weights.lock` (verify: `shasum -a 256 -c` from the model dir).

**Next:** Phase 0 — parse config.json, print architecture summary (mind the
explicit `head_dim` gotcha), then the NumPy forward pass component by component.
