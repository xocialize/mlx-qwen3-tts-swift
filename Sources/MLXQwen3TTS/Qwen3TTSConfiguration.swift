import Foundation
import MLXToolKit

/// Init-time configuration for `Qwen3TTSPackage` (C9): which checkpoint to load and the
/// default synthesis language. Per-request voice/instruct/language ride the `TTSRequest`
/// envelope, not here.
///
/// **FootprintConfigured (1.14 efficiency contract).** The footprint varies along THREE axes —
/// variant × size × quant — so a quant-keyed `QuantFootprint` (and `QuantConfigured`) can't express
/// it: 0.6B-8bit and 1.7B-8bit are the same quant but very different working sets. This is the
/// per-config-hint case (same as mlx-qwen-llm-swift, BiRefNet matting). The hints declare the
/// **selected** checkpoint's split — persistent weights + the measured autoregressive transient — so
/// the governor charges it precisely instead of the static manifest's default-variant figure.
public struct Qwen3TTSConfiguration: PackageConfiguration, ModelStorable, FootprintConfigured {
    /// Which mlx-community checkpoint to load (variant × size × quant).
    public var checkpoint: Qwen3TTSCheckpoint
    /// Language used when a request doesn't specify one in `metaData["language"]`.
    public var defaultLanguage: String
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// core's default cache. Excluded from `Codable` (a URL is environment-specific, not
    /// portable config).
    public var modelsRootDirectory: URL?

    public init(checkpoint: Qwen3TTSCheckpoint = .default,
                defaultLanguage: String = "english",
                modelsRootDirectory: URL? = nil) {
        self.checkpoint = checkpoint
        self.defaultLanguage = defaultLanguage
        self.modelsRootDirectory = modelsRootDirectory
    }

    // MARK: FootprintConfigured — the selected (variant × size × quant) checkpoint's split footprint

    /// Persistent weights floor of the selected checkpoint (quantized Talker + CodePredictor +
    /// bf16 codec stack + headroom).
    public var residentBytesHint: UInt64? { checkpoint.estimatedResidentBytes }

    /// Measured autoregressive transient (Talker generation scratch) at the documented synth
    /// envelope, scaled by model size. See `Qwen3TTSCheckpoint.estimatedPeakActivationBytes`.
    public var peakActivationBytesHint: UInt64? { checkpoint.estimatedPeakActivationBytes }

    private enum CodingKeys: String, CodingKey {
        case checkpoint, defaultLanguage
    }
}
