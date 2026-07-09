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
    /// Explicit checkpoint directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Explicit shared tokenizer/codec directory (dev escape hatch).
    public var tokenizerDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (a URL is environment-specific, not portable
    /// config).
    public var modelsRootDirectory: URL?

    public init(checkpoint: Qwen3TTSCheckpoint = .default,
                defaultLanguage: String = "english",
                modelDirectory: URL? = nil,
                tokenizerDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.checkpoint = checkpoint
        self.defaultLanguage = defaultLanguage
        self.modelDirectory = modelDirectory
        self.tokenizerDirectory = tokenizerDirectory
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

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Qwen3TTSConfiguration: WeightSourcing {
    /// Tokenizer sidecars the wrapper needs alongside the checkpoint (the cloning engine reads
    /// `vocab.json`; the core reads all three). Today the core's `HuggingFaceDownloader` fetches
    /// them via `additionalFiles`.
    static let modelSidecarFiles = ["vocab.json", "merges.txt", "tokenizer_config.json"]
    /// The checkpoint download set: config + weights (sharded or single, incl. the Base
    /// variant's `speech_tokenizer/model.safetensors` codec encoder) + sidecars — exactly the
    /// globs the core's downloader builds.
    static let modelGlobs = ["config.json", "*.safetensors", "model.safetensors.index.json"]
        + modelSidecarFiles
    /// The shared tokenizer/codec repo set (config + codec weights).
    static let tokenizerGlobs = ["config.json", "*.safetensors", "model.safetensors.index.json"]

    /// The declaration is **quant-VARIANT**: the checkpoint (variant × size × quant) computes
    /// the model repo, unlike IndexTTS2's quant-invariant set. The shared tokenizer/codec repo
    /// is common to all checkpoints.
    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "model", repo: checkpoint.repoID, matching: Self.modelGlobs),
            WeightSource(role: "tokenizer", repo: Qwen3TTSCheckpoint.tokenizerRepoID,
                         matching: Self.tokenizerGlobs),
        ]
    }

    /// A directory "has weights" when any safetensors file is present (the core's own
    /// `weightsExist` probe — shard names vary across the 25-checkpoint catalog).
    private static func hasSafetensors(_ dir: URL) -> Bool {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    private static func hasModelSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: dir.appending(path: "config.json").path)
            && fm.fileExists(atPath: dir.appending(path: "vocab.json").path)
            && hasSafetensors(dir)
    }

    private static func hasTokenizerSnapshot(_ dir: URL) -> Bool {
        FileManager.default.fileExists(atPath: dir.appending(path: "config.json").path)
            && hasSafetensors(dir)
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let store = ModelStore(root: storeRoot)
        return weightSources.filter { source in
            switch source.role {
            case "model":
                if let dir = modelDirectory, Self.hasModelSnapshot(dir) { return false }
                if let dir = store.directory(for: source.repo), Self.hasModelSnapshot(dir) {
                    return false
                }
                return true
            default:  // tokenizer
                if let dir = tokenizerDirectory, Self.hasTokenizerSnapshot(dir) { return false }
                if let dir = store.directory(for: source.repo), Self.hasTokenizerSnapshot(dir) {
                    return false
                }
                return true
            }
        }
    }

    /// The configuration with nil directories resolved to the store layout — what `load()`
    /// uses AFTER materialization. Explicit directories always win.
    public func resolved(storeRoot: URL?) -> Qwen3TTSConfiguration {
        let store = ModelStore(root: storeRoot)
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = store.directory(for: checkpoint.repoID)
        }
        if cfg.tokenizerDirectory == nil {
            cfg.tokenizerDirectory = store.directory(for: Qwen3TTSCheckpoint.tokenizerRepoID)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension Qwen3TTSConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved snapshot directories (checkpoint + shared codec); the prewarmer scans
        // them recursively and skips them when absent (first launch).
        let r = resolved(storeRoot: modelsRootDirectory)
        return [r.modelDirectory, r.tokenizerDirectory].compactMap { $0 }
    }
}
