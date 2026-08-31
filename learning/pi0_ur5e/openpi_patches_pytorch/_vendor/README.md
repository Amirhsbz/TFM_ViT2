# Vendored FTP1 files

These three files are copied, read-only reference material from the `ftp1-policy` repo
(`src/openpi/models_pytorch/`), at commit `076cd9a6c9e32629549655ddcf795eb86ae2f7c4` (the commit
that introduced `transformers_replace`, which these depend on). They are vendored here (rather
than requiring a live `ftp1-policy` checkout at install time) so `scripts/install_openpi_pytorch_patch.py`
is self-contained and re-runnable against a fresh `$OPENPI_ROOT`.

- `ftp1_attention_masks.py` — copied verbatim.
- `ftp1_gemma_pytorch.py` — copied verbatim.
- `t3_tactile_encoder.py` — copied with one import change: its dependency on ftp1-policy's
  `ftp1_model_config.py` (the full FTP1 model config, out of scope for this port) was replaced
  with a lazy import of the two T3-checkpoint constants from `../haptile_tactile_encoder.py`
  (which now hosts them), to avoid pulling in that unported file. See
  `../docs/ftp1_tactile_expert_port.md` for the full rationale.

Do not hand-edit these files; if `ftp1-policy` updates, re-copy and re-apply the same import fix.
