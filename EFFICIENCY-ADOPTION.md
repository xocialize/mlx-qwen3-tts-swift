# Efficiency Adoption Brief — `mlx-qwen3-tts-swift` (Qwen3-TTS, `tts`)

> **For a session-specific agent.** Adopt the engine 1.14 efficiency contract (engine 0.15.0). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"** — the autoregressive prefill lesson applies) + references/memory-harness.md. Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXQwen3TTS` (`Qwen3TTSPackage`). Components: **Talker (autoregressive)** + **CodePredictor** +
  **codec stack** (bf16). Capability `tts`. **Multi-checkpoint:** variant (Base / CustomVoice /
  VoiceDesign + ICL cloning) × size (1.7B…) × quant — `Qwen3TTSCheckpoint` computes `residentBytes` per
  checkpoint (`talkerBytes(quant) + codecStackBytes + headroom`).
- **Footprint today:** computed per checkpoint, but **flat** (weights + headroom; no transient split).
- Engine pinned `from: "0.3.0"`.

## The shape (two relevant lessons)
1. **Autoregressive Talker** → the transient is generation scratch + KV/cache that scales with the
   **generated audio-token length** (the Qwen-LLM "measure the prefill/decode transient, don't derive it"
   lesson — verify against a real synth run, the talker is a different arch than a text LM).
2. **Multi-component** (Talker + CodePredictor + codec) → a possible **per-stage** angle: the codec stack
   / CodePredictor usage pattern may allow staging, but TTS talker+codec are often interleaved — verify
   before assuming an evict point (don't force one if they're co-active).

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.15.0 | **P0** |
| 1. Split footprint | ❌ | per-checkpoint flat residentBytes; autoregressive transient unaccounted | **P1** |
| 2. Per-stage evict | 🟡 verify | Talker/CodePredictor/codec — check if any component is idle long enough to evict | P2 (verify) |
| 3. mmap/lazy | 🟡 verify | confirm lazy/mmap load (floor ≈ on-disk) | note |
| 4. BudgetAware | ➖ | quant config-chosen | defer |

## Plan
- **P0:** `swift package update` → 0.15.0; build + fix any drift.
- **P1 (the lever):** conform the config to **`FootprintConfigured`** (variant×size×quant can't be
  expressed by `quant` alone — same per-config-hint case as Qwen-LLM). `residentBytesHint` = the
  checkpoint's weights (Talker + CodePredictor + codec); `peakActivationBytesHint` = the **measured**
  generation transient at a documented synth envelope (a representative sentence → N audio tokens). Keep
  the existing per-checkpoint `residentBytes` derivation as the weights basis. **Measure the transient,
  don't derive it.**
- **P2:** only if a component is genuinely idle for a phase — verify the Talker→CodePredictor→codec flow;
  if they interleave per audio frame, P2 is N/A (note it). Don't force an evict that re-loads per frame.
- **P3** mmap (note). **P4** defer.
- **Measure** via the package's own smoke/CLI target (`xcodebuild`): load checkpoint → weights floor → a
  representative synth → peak. Light (audio gen). Document the synth envelope.

## Definition of done
- [ ] engine 0.15.0; `FootprintConfigured` (weights hint + measured transient @ documented synth envelope).
- [ ] P2 decided (staged or N/A-with-reason); mmap noted; BudgetAware deferred.
- [ ] Smoke green (valid non-silent WAV — guard against the silent-stem failure class); split recorded.
- [ ] Registry: mlx-qwen3-tts-swift row Eff ⬜→✅, Eng→0.15.0.

## Report back
Flat→split, the measured autoregressive transient + synth envelope, P2 decision, drift, effort, SHAs.
STAY IN SCOPE — four-lever adoption + brief + registry row only; stop-and-report if bigger.
