// CancellationTests.swift — Qwen3-TTS through the engine's CAN gate (offline, no MLX kernels):
// the motivating package of the gate — the Talker loop originally shipped with no checkpoints
// and nothing caught it. CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires
// before notLoaded validation or weights); CAN-3 is the document of record for the checkpoint
// cadence: the core's Talker loops bail per generated codec frame (E3 `Task.isCancelled` break
// in generateWithCodePredictor / batch / cloning GenerationLoop) and the wrapper's post-
// synthesis `try Task.checkCancellation()` rethrows the CancellationError unchanged.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXQwen3TTS

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = Qwen3TTSPackage(configuration: Qwen3TTSConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: TTSRequest(text: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // tts is a long-run capability — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Qwen3TTSPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: Qwen3TTSPackage.manifest,
            posture: .cadence([
                // The autoregressive Talker loop checks Task.isCancelled once per generated
                // codec frame (12.5 Hz — E3 in the qwen3-tts-mlx-swift core, all four loops:
                // synthesize / stream / batch / ICL-cloning GenerationLoop).
                .init(phase: .generate, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
