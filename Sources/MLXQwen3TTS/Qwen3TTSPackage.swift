import Foundation
import MLX
import MLXToolKit
import Qwen3TTS
import AudioCommon
import Qwen3TTSCloning

/// Errors specific to the Qwen3-TTS package boundary.
public enum Qwen3TTSError: Error, Equatable {
    /// Weight sources are missing and there is no store root (or resolved directory) to
    /// materialize into.
    case missingWeights(String)
}

/// Qwen3-TTS exposing the canonical `tts` surface: base synthesis, CustomVoice preset
/// speakers (`.named`), and voice cloning from reference audio (`.referenceAudio`) — x-vector
/// by default, full ICL (~0.89 speaker similarity) when `metaData["referenceText"]` carries
/// the reference transcript.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `Qwen3TTSConfiguration`, pages
/// weights in with `load()` (mlx-community checkpoint + the shared Qwen tokenizer/codec repo,
/// materialized under the engine's models root when set), drives `run(_:)`, and reclaims with
/// `unload()`. Returns the canonical `Audio` (.wav, 24 kHz mono) artifact.
///
/// The reference transcript rides the canonical `TTSRequest.referenceTranscript` field
/// (contract 1.1.0 — promoted from `metaData` when this wrap surfaced the gap).
/// `metaData` keys (package-specific, C5): `language` (synthesis language, default from
/// configuration), `instruct` (CustomVoice emotion instruct / VoiceDesign voice description),
/// `seed` (Int — pins the sampled voice for character consistency; with the VoiceDesign
/// checkpoint this is the Voice Library "Character" seed, stable across chunks of a take).
@InferenceActor
public final class Qwen3TTSPackage: ModelPackage {
    public typealias Configuration = Qwen3TTSConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Qwen3-TTS weights are Apache-2.0 (Alibaba/Qwen, mlx-community conversions);
            // the port (qwen3-tts-mlx-swift, derived from soniqo/speech-swift) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: Qwen3TTSCheckpoint.default.repoID, revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Footprints for the default 1.7B Base checkpoint across the full quant catalog
                // (int5/int6 added in contract 1.1.0). Split per contract 1.14: persistent weights
                // (quantized Talker+CodePredictor + bf16 codec stack + headroom) in `residentBytes`,
                // and the MEASURED autoregressive Talker transient in `peakActivationBytes` (the
                // engine reserves a single transient across residents — see memory-harness.md).
                // The precise per-configured-checkpoint split rides `FootprintConfigured` on the
                // configuration (variant × size × quant); these static rows are the default-variant
                // fallback for a registration that doesn't carry a config hint.
                footprints: Qwen3TTSCheckpointQuant.allCases.map { quant in
                    let ckpt = Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: quant)
                    return QuantFootprint(
                        quant: quant.toolKitQuant,
                        residentBytes: ckpt.estimatedResidentBytes,
                        peakActivationBytes: ckpt.estimatedPeakActivationBytes)
                },
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                TTSContract.descriptor(
                    name: "qwen3-tts",
                    summary: "Qwen3-TTS multilingual text-to-speech with preset speakers and "
                        + "reference-audio voice cloning (.wav, 24 kHz).",
                    modes: [.neutral, .expressive]
                )
            ]
        )
    }

    private let configuration: Configuration
    // Non-Sendable core types from the Swift-5-mode Qwen3TTS target. The engine serializes all
    // lifecycle calls on InferenceActor, so there's no real concurrency (see Package.swift).
    private var model: Qwen3TTSModel?
    private var tokenizer: Qwen3Tokenizer?
    // Voice-clone prompt reuse (E1): building an ICL prompt re-runs ECAPA speaker
    // embedding + native codec encode + tokenization of the reference. Long-form
    // synthesis (e.g. TTSOrchestratorKit anchor-to-first-segment) sends the *same*
    // reference for every chunk, so we memoize the prepared prompt keyed by the
    // (reference bytes, transcript, language) triple and skip that work on reuse.
    // Safe to hold across calls: InferenceActor serializes run(), and VoiceClonePrompt
    // is read-only once built.
    private var cloningEngine: VoiceCloningEngine?
    private var cachedClonePrompt: (key: Int, prompt: VoiceClonePrompt)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        // Auto-materialize missing sources into the engine store (dir-less configs only; explicit
        // directories never touch the network). The download code is the core's existing
        // HuggingFaceDownloader (it already lands snapshots in the store layout,
        // <root>/<org>/<name>); the WeightSourcing delta is the declared missing-set + progress
        // forwarding — source i of n maps onto fraction [i/n, (i+1)/n) so the engine's
        // PreparationMonitor shows one monotonic `.downloading` bar.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw Qwen3TTSError.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            for (index, source) in missing.enumerated() {
                let base = Double(index) / Double(missing.count)
                let span = 1.0 / Double(missing.count)
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: source.repo,
                    to: storeRoot.appendingPathComponent(source.repo),
                    additionalFiles: source.role == "model"
                        ? Qwen3TTSConfiguration.modelSidecarFiles : [],
                    progressHandler: { fraction in
                        WeightDownloadProgress.report(fraction: base + span * fraction)
                    })
                try Task.checkCancellation()
            }
        }
        try Task.checkCancellation()

        // Load from the resolved (explicit-or-store) directories.
        let resolved = configuration.resolved(storeRoot: storeRoot)
        guard let modelDir = resolved.modelDirectory,
              let tokenizerDir = resolved.tokenizerDirectory else {
            throw Qwen3TTSError.missingWeights("unresolved weight directories (no store root)")
        }
        let loaded = try await Qwen3TTSModel.fromPretrained(
            modelId: modelDir.path, tokenizerModelId: tokenizerDir.path)

        // A wrapper-owned tokenizer instance for the cloning engine (the model's is private).
        let vocabURL = modelDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            let tok = Qwen3Tokenizer()
            try tok.load(from: vocabURL)
            tokenizer = tok
        }

        model = loaded
    }

    public func unload() async {
        model = nil
        tokenizer = nil
        cloningEngine = nil
        cachedClonePrompt = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let language = tts.metaData.stringValue("language") ?? configuration.defaultLanguage
        let instruct = tts.metaData.stringValue("instruct")

        // Sampling: the VoiceDesign checkpoint wants its creative preset; everything else uses
        // defaults. `metaData["seed"]` pins the voice (Voice Library "Character" consistency) —
        // the core re-seeds the RNG before generation, so the same seed yields the same timbre
        // across chunks of a long take.
        var sampling = (configuration.checkpoint.variant == .voiceDesign)
            ? SamplingConfig.voiceDesign : SamplingConfig()
        if let seed = tts.metaData.intValue("seed") {
            // Clamp to 32-bit. Large 64-bit seeds (≳5×10^18) produce Gumbel-noise patterns
            // that never favor EOS → runaway/infinite generation (verified in prior dub work;
            // the core's `speakerSeed` truncates for the same reason). Guard at the engine
            // boundary so no supplied seed can reach MLXRandom.seed unclamped.
            sampling.seed = UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF
        }

        let samples: [Float]
        switch tts.voice.selection {
        case .auto:
            samples = model.synthesize(
                text: tts.text, language: language, instruct: instruct, sampling: sampling)

        case .named(let speaker):
            samples = model.synthesize(
                text: tts.text, language: language, speaker: speaker, instruct: instruct,
                sampling: sampling)

        case .referenceAudio(let reference):
            if let referenceText = tts.referenceTranscript,
               !referenceText.isEmpty, let tokenizer {
                // Full ICL cloning: reference codes + transcript + speaker embedding.
                let engine = cloningEngine ?? VoiceCloningEngine(model: model, tokenizer: tokenizer)
                cloningEngine = engine

                // Reuse the prepared prompt when the same reference repeats (E1).
                let key = Self.cloneKey(data: reference.data, text: referenceText, language: language)
                let prompt: VoiceClonePrompt
                if let cached = cachedClonePrompt, cached.key == key {
                    prompt = cached.prompt
                } else {
                    let (refSamples, refRate) = try Self.decodeWAV(reference)
                    prompt = try engine.createPrompt(
                        referenceAudio: refSamples, referenceText: referenceText,
                        sampleRate: refRate, language: language)
                    cachedClonePrompt = (key, prompt)
                }
                try Task.checkCancellation()
                samples = try engine.synthesize(text: tts.text, prompt: prompt, mode: .icl)
            } else {
                // x-vector cloning: speaker embedding only (no transcript required).
                let (refSamples, refRate) = try Self.decodeWAV(reference)
                samples = model.synthesizeWithVoiceClone(
                    text: tts.text, referenceAudio: refSamples,
                    referenceSampleRate: refRate, language: language)
            }
        }

        try Task.checkCancellation()
        let wav = Self.encodeWAV16(samples: samples, sampleRate: 24_000)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: 24_000, channels: 1))
    }

    // MARK: - Voice-clone prompt cache key (E1)

    /// Cache key for a prepared ICL prompt: the reference clip bytes + transcript +
    /// synthesis language. In-memory only (Hasher is per-process seeded), which is all
    /// we need — reuse happens within a single long-form run.
    nonisolated static func cloneKey(data: Data, text: String, language: String) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        hasher.combine(text)
        hasher.combine(language)
        return hasher.finalize()
    }

    // MARK: - Canonical Audio codec (serialized round-trip form, C3)

    /// Decodes a canonical `Audio` (.wav) artifact into mono float samples + sample rate.
    nonisolated static func decodeWAV(_ audio: Audio) throws -> (samples: [Float], sampleRate: Int) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-qwen3-tts-ref-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        return try AudioFileLoader.loadWAV(url: tmp)
    }

    /// Encodes mono float samples as a 16-bit PCM WAV (broadly playable) in memory.
    nonisolated static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        let channels = 1
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = samples.count * blockAlign

        var data = Data(capacity: 44 + dataSize)
        func ascii(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        ascii("RIFF"); u32(UInt32(36 + dataSize)); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1) // PCM
        u16(UInt16(channels)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(UInt16(bitsPerSample))
        ascii("data"); u32(UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var le = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        return data
    }
}

extension Qwen3TTSPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(Qwen3TTSPackage.self)
    }
}

extension MetaData {
    /// Convenience: read a string-valued metaData key.
    func stringValue(_ key: String) -> String? {
        if case .string(let value)? = self[key] { return value }
        return nil
    }

    /// Convenience: read an int-valued metaData key (e.g. a voice seed).
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }
}
