# Haptile tactile-expert model — development summary

This is a narrative summary of how the model itself evolved, stage by stage, from the original
Haptile (no learned tactile component at all) to the current tactile-expert model (pretrained,
partially LoRA-adapted backbone; pretrained, fully fine-tuned tactile encoder; fully-trained
tactile expert). For full implementation detail, see `ftp1_tactile_expert_port.md` (Part 2) and
`../../docs/tactile_raw_image_pipeline.md` (Part 1).

## Stage 0 — Original Haptile (no tactile encoder or expert at all)

Tactile entered the model as a fixed, untrained random projection of the raw pixels, concatenated
onto `observation.state`. There was no tactile-specific neural network component whatsoever —
nothing to train, nothing to initialize, nothing to freeze. The model only ever saw a wider state
vector.

## Stage 1 — Tactile-expert architecture introduced (before this conversation)

The real architectural leap: a learned ViT tactile encoder + a dedicated tactile-expert
transformer branch, cross-attended by the action expert. First time tactile pixels were processed
by an actual trainable network. But at this point:

- **Tactile encoder**: randomly initialized. The code had a `load_t3_pretrained_checkpoint`
  option already built into the underlying ViT class, but it was never exposed or turned on.
- **VLM + action-expert backbone**: also randomly initialized, fully trained, hardcoded to the
  pi0.5 convention (discrete state folded into the prompt) — a mismatch with the project's actual
  established convention that went unnoticed until this session.
- Two latent bugs shipped with it: `discrete_state_input` defaulting wrong (would have left the
  model blind to robot state) and a core assert that would have blocked the transform pipeline
  from building at all.

## Stage 2 — This conversation's decisions

1. **Verified the pipeline actually assembles** (a real dry run), which surfaced and fixed both
   latent bugs above.
2. **Pi0 vs pi0.5 reconciled**: switched the default to pi0 (continuous state, its own suffix
   token) to match what every other task in this project actually uses — pi0.5 had never been
   validated as the right choice, just inherited from an unrelated config's unset-env-var default.
3. **Tactile encoder — pretrained initialization added, kept fully trainable**: wired up and
   fixed the T3 checkpoint loading (a real size-class bug, `t3_large` vs `t3_medium`). Now starts
   from real pretrained tactile representations and fine-tunes fully from there — same
   "pretrained, no freezing" recipe as before, just no longer starting from scratch.
4. **VLM + action-expert backbone — pretrained initialization + LoRA added**: converted a real
   `pi0_base` checkpoint, built a tolerant loader for it, and added LoRA (via `peft`) that freezes
   the backbone and trains only small adapters — reusing config naming that already existed but
   had been doing nothing. Explicitly gated so LoRA only activates when pretrained weights are
   actually loaded, never applied to a randomly-initialized backbone.
5. **Tactile expert — deliberately left fully trained from scratch throughout**, at explicit
   confirmation. No pretrained tactile-expert checkpoint exists anywhere to seed it from, so full
   training was always the only sensible choice for this one component.

## The throughline

Every component now falls into exactly one of two buckets, by deliberate choice rather than
default:

| Bucket | Components |
|---|---|
| **Pretrained-and-adapted** | VLM + action-expert (frozen backbone + LoRA adapters); tactile encoder (fully fine-tuned from pretrained T3 weights) |
| **Trained-from-scratch** | Tactile expert (no pretrained analogue exists to adapt from) |

Stage 1 had no such distinction — everything was either fixed-and-untrainable (Stage 0's random
projection) or randomly-initialized-and-fully-trained, with no pretrained signal anywhere in the
tactile-specific parts of the model.
