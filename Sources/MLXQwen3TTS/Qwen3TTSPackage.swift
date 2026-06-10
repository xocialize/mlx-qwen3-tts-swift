import Foundation
import MLXToolKit
import Qwen3TTS
import AudioCommon
import Qwen3TTSCloning

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
/// `metaData` keys (package-specific, C5): `language` (synthesis language, default from
/// configuration), `instruct` (CustomVoice emotion instruct / VoiceDesign voice description),
/// `referenceText` (reference transcript — upgrades `.referenceAudio` from x-vector to ICL).
/// Note: a reference *transcript* is arguably should-be-canonical for ICL-capable TTS
/// (VoxCPM2 needs the same lever) — candidate for a TTS schema revision; tracked as a gap.
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
                // Footprints for the default 1.7B Base checkpoint. Quantized Talker+CodePredictor
                // plus the bf16 codec stack (decoder/encoder/speaker encoder) and headroom.
                // Static-manifest caveat (same as mlx-qwen-llm-swift): admission gates on the
                // default variant; per-configured-checkpoint gating is a follow-up.
                footprints: [
                    QuantFootprint(
                        quant: .int4,
                        residentBytes: Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .q4)
                            .estimatedResidentBytes),
                    QuantFootprint(
                        quant: .int8,
                        residentBytes: Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .q8)
                            .estimatedResidentBytes),
                    QuantFootprint(
                        quant: .bf16,
                        residentBytes: Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .bf16)
                            .estimatedResidentBytes),
                ],
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

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        // Resolve weight directories. With a models root set, materialize both repos under it
        // (<root>/<namespace>/<name>); otherwise pass repo ids through to the core's default cache.
        let modelRef: String
        let tokenizerRef: String
        if let root = configuration.modelsRootDirectory {
            modelRef = try await Self.materialize(
                repoID: configuration.checkpoint.repoID, under: root,
                additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"])
            tokenizerRef = try await Self.materialize(
                repoID: Qwen3TTSCheckpoint.tokenizerRepoID, under: root)
        } else {
            modelRef = configuration.checkpoint.repoID
            tokenizerRef = Qwen3TTSCheckpoint.tokenizerRepoID
        }

        try Task.checkCancellation()
        let loaded = try await Qwen3TTSModel.fromPretrained(
            modelId: modelRef, tokenizerModelId: tokenizerRef)

        // A wrapper-owned tokenizer instance for the cloning engine (the model's is private).
        let modelDir: URL
        if FileManager.default.fileExists(atPath: modelRef) {
            modelDir = URL(fileURLWithPath: modelRef)
        } else {
            modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelRef)
        }
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
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let language = tts.metaData.stringValue("language") ?? configuration.defaultLanguage
        let instruct = tts.metaData.stringValue("instruct")

        let samples: [Float]
        switch tts.voice.selection {
        case .auto:
            samples = model.synthesize(text: tts.text, language: language, instruct: instruct)

        case .named(let speaker):
            samples = model.synthesize(
                text: tts.text, language: language, speaker: speaker, instruct: instruct)

        case .referenceAudio(let reference):
            let (refSamples, refRate) = try Self.decodeWAV(reference)
            if let referenceText = tts.metaData.stringValue("referenceText"),
               !referenceText.isEmpty, let tokenizer {
                // Full ICL cloning: reference codes + transcript + speaker embedding.
                let cloning = VoiceCloningEngine(model: model, tokenizer: tokenizer)
                let prompt = try cloning.createPrompt(
                    referenceAudio: refSamples, referenceText: referenceText,
                    sampleRate: refRate, language: language)
                try Task.checkCancellation()
                samples = try cloning.synthesize(text: tts.text, prompt: prompt, mode: .icl)
            } else {
                // x-vector cloning: speaker embedding only (no transcript required).
                samples = model.synthesizeWithVoiceClone(
                    text: tts.text, referenceAudio: refSamples,
                    referenceSampleRate: refRate, language: language)
            }
        }

        try Task.checkCancellation()
        let wav = Self.encodeWAV16(samples: samples, sampleRate: 24_000)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: 24_000, channels: 1))
    }

    // MARK: - Weights materialization

    /// Download a repo's weights under `<root>/<namespace>/<name>` (idempotent) and return the
    /// local path, which the core's `fromPretrained` accepts in place of a repo id.
    private nonisolated static func materialize(
        repoID: String, under root: URL, additionalFiles: [String] = []
    ) async throws -> String {
        let dir = root.appendingPathComponent(repoID)
        if !HuggingFaceDownloader.weightsExist(in: dir) {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: repoID, to: dir, additionalFiles: additionalFiles)
        }
        return dir.path
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
}
