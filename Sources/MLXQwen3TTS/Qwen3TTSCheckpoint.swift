import Foundation
import MLXToolKit

/// The Qwen3-TTS model variant — which checkpoint family to load.
public enum Qwen3TTSVariant: String, Sendable, Codable, CaseIterable {
    /// Standard synthesis + voice cloning (speaker encoder + codec encoder included).
    case base = "Base"
    /// 9 preset speakers with emotion instructs (no speaker encoder — no cloning).
    case customVoice = "CustomVoice"
    /// Voice design via natural-language description (1.7B only).
    case voiceDesign = "VoiceDesign"
}

/// Model size. 0.6B is the broad-eligibility option; 1.7B is the quality option.
public enum Qwen3TTSSize: String, Sendable, Codable, CaseIterable {
    case s0_6B = "0.6B"
    case s1_7B = "1.7B"

    /// Talker + CodePredictor parameter count (approximate, for footprint estimates).
    var parameterCount: UInt64 {
        switch self {
        case .s0_6B: return 600_000_000
        case .s1_7B: return 1_700_000_000
        }
    }
}

/// Checkpoint quantization, named by the mlx-community repo suffix.
public enum Qwen3TTSCheckpointQuant: String, Sendable, Codable, CaseIterable {
    case q4 = "4bit"
    case q5 = "5bit"
    case q6 = "6bit"
    case q8 = "8bit"
    case bf16 = "bf16"

    /// Approximate bytes per parameter (weights + per-group scales/biases for affine quants).
    var bytesPerParameter: Double {
        switch self {
        case .q4: return 0.56
        case .q5: return 0.69
        case .q6: return 0.81
        case .q8: return 1.06
        case .bf16: return 2.0
        }
    }

    /// The matching `MLXToolKit.Quant` (int5/int6 added in contract 1.1.0).
    var toolKitQuant: Quant {
        switch self {
        case .q4: return .int4
        case .q5: return .int5
        case .q6: return .int6
        case .q8: return .int8
        case .bf16: return .bf16
        }
    }
}

/// One checkpoint in the mlx-community Qwen3-TTS catalog = variant × size × quant.
/// The repo id is computed, never hardcoded — same pattern as `QwenModel` in
/// mlx-qwen-llm-swift. We consume the community conversions directly; no weight duplication.
public struct Qwen3TTSCheckpoint: Sendable, Codable, Equatable {
    public var variant: Qwen3TTSVariant
    public var size: Qwen3TTSSize
    public var quant: Qwen3TTSCheckpointQuant

    public init(variant: Qwen3TTSVariant, size: Qwen3TTSSize, quant: Qwen3TTSCheckpointQuant) {
        self.variant = variant
        self.size = size
        self.quant = quant
    }

    /// Default: 1.7B Base 8-bit — the quality/footprint balance point. The earlier "4-bit is
    /// garbled" finding was against a third-party 0.6B conversion; the 8-bit default still
    /// needs its live listen test (see APP-VALIDATION.md) with bf16 as the known-good fallback.
    public static let `default` = Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .q8)

    /// `mlx-community/Qwen3-TTS-12Hz-<size>-<Variant>-<quant>`
    public var repoID: String {
        "mlx-community/Qwen3-TTS-12Hz-\(size.rawValue)-\(variant.rawValue)-\(quant.rawValue)"
    }

    /// The shared text-tokenizer + speech-codec (decoder/encoder) weights repo, common to all
    /// checkpoints — downloaded once.
    public static let tokenizerRepoID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    /// VoiceDesign was only released at 1.7B; everything else exists in all quants.
    public var isPublished: Bool {
        !(variant == .voiceDesign && size == .s0_6B)
    }

    /// The full published catalog (25 checkpoints as of 2026-06).
    public static var allPublished: [Qwen3TTSCheckpoint] {
        Qwen3TTSVariant.allCases.flatMap { variant in
            Qwen3TTSSize.allCases.flatMap { size in
                Qwen3TTSCheckpointQuant.allCases.compactMap { quant in
                    let c = Qwen3TTSCheckpoint(variant: variant, size: size, quant: quant)
                    return c.isPublished ? c : nil
                }
            }
        }
    }

    /// Estimated resident bytes: quantized Talker + CodePredictor, plus the bf16 codec stack
    /// (speech tokenizer decoder + encoder + speaker encoder, ~0.7 GB) and runtime headroom.
    /// This is the persistent **weights floor** (contract 1.14 `residentBytes`) — the transient
    /// generation scratch is split out into `peakActivationBytes`, NOT folded in here.
    public var estimatedResidentBytes: UInt64 {
        let talkerBytes = Double(size.parameterCount) * quant.bytesPerParameter
        let codecStackBytes = 700_000_000.0
        let headroom = 500_000_000.0
        return UInt64(talkerBytes + codecStackBytes + headroom)
    }

    /// The **measured** autoregressive transient (contract 1.14 `peakActivationBytes`) — the Talker
    /// generation scratch (KV/cache + decode buffers) live only during a forward pass, on top of the
    /// resident weights. The Talker is autoregressive, so this is generation scratch that scales with
    /// the generated audio-token length; it is largely quant-independent (it's the activation graph,
    /// not the weights) and scales with model size.
    ///
    /// **Measured** (the "measure the transient, don't derive it" lesson) on **1.7B Base 8bit** via
    /// the package's own gated bench (`Qwen3TTSMemoryReport`, Q3T_MEM=1) at a documented synth
    /// envelope — zero-shot, ~168 chars → ~10 s of 24 kHz audio (the 12 Hz codec, the representative
    /// chunk a long-form orchestrator sends): resident floor 2.65 GB · worst peak 6.66 GB ⇒
    /// transient ≈ 4.0 GB. Declared with ~20% headroom (memory-harness.md) → ~4.8 GB at 1.7B, scaled
    /// linearly by parameter count for the 0.6B tier (activations track sequence × hidden, ~size).
    public var estimatedPeakActivationBytes: UInt64 {
        // Measured base: 1.7B → ~4.0 GB transient + 20% headroom ≈ 4.8 GB.
        let base1_7B = 4_800_000_000.0
        let scale = Double(size.parameterCount) / 1_700_000_000.0
        return UInt64(base1_7B * scale)
    }
}
