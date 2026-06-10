import Foundation
import MLXToolKit

/// Init-time configuration for `Qwen3TTSPackage` (C9): which checkpoint to load and the
/// default synthesis language. Per-request voice/instruct/language ride the `TTSRequest`
/// envelope, not here.
public struct Qwen3TTSConfiguration: PackageConfiguration, ModelStorable {
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

    private enum CodingKeys: String, CodingKey {
        case checkpoint, defaultLanguage
    }
}
