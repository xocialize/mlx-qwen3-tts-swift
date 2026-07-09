// MaterializationTests.swift — Qwen3-TTS through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. The declaration is QUANT-VARIANT (the checkpoint's
// variant × size × quant computes the model repo — unlike IndexTTS2's quant-invariant set),
// so the gate runs across the full published catalog. The shared tokenizer/codec repo is a
// second, checkpoint-invariant source.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXQwen3TTS

final class MaterializationTests: XCTestCase {

    /// Temp dirs holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDirs() throws -> (model: URL, tokenizer: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "qwen3tts-mat-\(UUID().uuidString)")
        let model = base.appending(path: "model")
        let tokenizer = base.appending(path: "tokenizer")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for f in ["config.json", "vocab.json", "model.safetensors"] {
            FileManager.default.createFile(atPath: model.appending(path: f).path, contents: Data([0]))
        }
        for f in ["config.json", "model.safetensors"] {
            FileManager.default.createFile(atPath: tokenizer.appending(path: f).path,
                                           contents: Data([0]))
        }
        return (model, tokenizer, { try? FileManager.default.removeItem(at: base) })
    }

    // MARK: - Engine MAT gate, across the published catalog

    func testMATGateAcrossPublishedCatalog() throws {
        let (model, tokenizer, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        for checkpoint in Qwen3TTSCheckpoint.allPublished {
            let report = MaterializationConformance.check(
                freshConfiguration: Qwen3TTSConfiguration(checkpoint: checkpoint),
                satisfiedConfiguration: Qwen3TTSConfiguration(
                    checkpoint: checkpoint, modelDirectory: model, tokenizerDirectory: tokenizer))
            XCTAssertTrue(report.passed, "checkpoint \(checkpoint.repoID): \(report.summary)")
        }
    }

    // MARK: - Source declaration shape (quant-variant + shared tokenizer)

    func testDeclarationFollowsTheCheckpoint() {
        let cfg = Qwen3TTSConfiguration()   // default: 1.7B Base 8bit
        let sources = cfg.weightSources
        XCTAssertEqual(sources.map(\.role), ["model", "tokenizer"])
        XCTAssertEqual(sources[0].repo, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
        XCTAssertEqual(sources[1].repo, "Qwen/Qwen3-TTS-Tokenizer-12Hz")
        // The model source carries the tokenizer sidecars the wrapper/cloning engine reads.
        XCTAssertTrue(sources[0].matching?.contains("vocab.json") ?? false)
        // A different variant/size/quant changes ONLY the model repo.
        let other = Qwen3TTSConfiguration(checkpoint:
            Qwen3TTSCheckpoint(variant: .voiceDesign, size: .s1_7B, quant: .bf16))
        XCTAssertEqual(other.weightSources[0].repo,
                       "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
        XCTAssertEqual(other.weightSources[1].repo, "Qwen/Qwen3-TTS-Tokenizer-12Hz")
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesPerSource() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "qwen3tts-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = Qwen3TTSConfiguration()
        // Empty store: both sources missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).map(\.role),
                       ["model", "tokenizer"])
        // Populate ONLY the shared tokenizer repo → the model source alone stays missing.
        let tokDir = root.appending(path: Qwen3TTSCheckpoint.tokenizerRepoID)
        try FileManager.default.createDirectory(at: tokDir, withIntermediateDirectories: true)
        for f in ["config.json", "model.safetensors"] {
            FileManager.default.createFile(atPath: tokDir.appending(path: f).path,
                                           contents: Data([0]))
        }
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).map(\.role), ["model"])
        // Populate the checkpoint repo → nothing missing; a DIFFERENT checkpoint still misses.
        let modelDir = root.appending(path: cfg.checkpoint.repoID)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for f in ["config.json", "vocab.json", "model.safetensors"] {
            FileManager.default.createFile(atPath: modelDir.appending(path: f).path,
                                           contents: Data([0]))
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        let otherQuant = Qwen3TTSConfiguration(checkpoint:
            Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .bf16))
        XCTAssertEqual(otherQuant.missingWeightSources(storeRoot: root).map(\.role), ["model"])
        // Resolution lands on the store layout; explicit dirs always win.
        let resolved = cfg.resolved(storeRoot: root)
        XCTAssertEqual(resolved.modelDirectory?.path, modelDir.path)
        XCTAssertEqual(resolved.tokenizerDirectory?.path, tokDir.path)
        let explicit = Qwen3TTSConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
        XCTAssertEqual(explicit.tokenizerDirectory?.path, tokDir.path)
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = Qwen3TTSConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(cfg.prewarmPaths.map(\.path), [
            root.appending(path: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit").path,
            root.appending(path: "Qwen/Qwen3-TTS-Tokenizer-12Hz").path,
        ])
    }

    func testCodableRoundTrip() throws {
        let cfg = Qwen3TTSConfiguration(
            checkpoint: Qwen3TTSCheckpoint(variant: .customVoice, size: .s0_6B, quant: .q4),
            defaultLanguage: "german",
            modelDirectory: URL(fileURLWithPath: "/x"),
            tokenizerDirectory: URL(fileURLWithPath: "/y"))
        let decoded = try JSONDecoder().decode(Qwen3TTSConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.checkpoint, cfg.checkpoint)
        XCTAssertEqual(decoded.defaultLanguage, "german")
        XCTAssertNil(decoded.modelDirectory)       // environment-specific, never encoded
        XCTAssertNil(decoded.tokenizerDirectory)
    }
}
