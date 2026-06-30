import XCTest
import Foundation
import MLX
import MLXToolKit
@testable import MLXQwen3TTS

/// Split-footprint memory bench for the 1.14 efficiency contract (gated: Q3T_MEM=1, needs the
/// Cmlx metallib bundle in .build/debug + the default checkpoint at DEV_ARCHIVE/models).
///
/// Measures the two halves the manifest must declare (see memory-harness.md):
///   1. Resident floor  — load + warmup synth + clearCache → activeMemory ≈ weights resident
///                        (quantized Talker + CodePredictor + bf16 codec stack).
///   2. Transient peak  — reset Memory.peakMemory, run a real synth with forced eval, read peakMemory;
///                        peakActivationBytes = worstPeak − resident floor. The Talker is
///                        autoregressive, so the transient is generation scratch + cache that scales
///                        with the generated audio-token length — MEASURE it, don't derive (the
///                        Qwen-LLM "measure the prefill/decode transient" lesson).
/// Guards the silent-output failure class (a silent stem reads −∞ dBFS) before trusting any number.
final class Qwen3TTSMemoryReport: XCTestCase {
    // A longer sentence → more generated audio tokens → a representative talker decode envelope.
    static let sentence =
        "The morning light spilled across the quiet harbor as the boats began to stir, and the "
        + "gulls wheeled overhead in the brightening sky above the slowly waking little town."

    func testMeasure_qwen3tts_1_7B_base_8bit() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["Q3T_MEM"] == "1", "Q3T_MEM=1")

        // Default checkpoint: 1.7B Base 8bit (the manifest's representative variant).
        let pkg = Qwen3TTSPackage(configuration: Qwen3TTSConfiguration(
            checkpoint: .default,
            modelsRootDirectory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models")))

        // --- load + warmup (compiles size-specific kernels, materializes weights) ---
        var t0 = Date()
        try await pkg.load()
        let loadSecs = -t0.timeIntervalSinceNow
        _ = try await pkg.run(TTSRequest(text: "Warm up.")) as! TTSResponse

        // --- resident floor: drop activations back to cache, then read active ---
        MLX.Memory.clearCache()
        let residentFloor = MLX.Memory.activeMemory

        // --- transient peak: rebase peak to current active (weights), run a real synth ---
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        t0 = Date()
        let resp = try await pkg.run(TTSRequest(text: Self.sentence)) as! TTSResponse
        let genSecs = -t0.timeIntervalSinceNow
        let worstPeak = MLX.Memory.peakMemory

        // --- silent-output guard: a valid non-silent 24 kHz WAV, not a dead stem ---
        let (samples, rate) = try Qwen3TTSPackage.decodeWAV(resp.audio)
        let peakAmp = samples.map { abs($0) }.max() ?? 0
        let rms = (samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(samples.count, 1))).squareRoot()
        XCTAssertTrue(peakAmp > 0.02 && peakAmp <= 1.0, "degenerate/silent level (peak \(peakAmp))")
        XCTAssertTrue(rms > 0.005 && rms < 0.4, "rms out of range (\(rms))")
        XCTAssertGreaterThan(samples.count, rate, "under 1s — implausible")

        let transient = max(0, worstPeak - residentFloor)
        let durSecs = Double(samples.count) / Double(rate)
        let gib = 1024.0 * 1024.0 * 1024.0
        print(String(
            format: """
            [Q3T-MEM] envelope: zero-shot, %d chars → %.1fs @ %d Hz (12 Hz codec)
            [Q3T-MEM] checkpoint: 1.7B Base 8bit
            [Q3T-MEM] load: %.1fs · gen: %.1fs
            [Q3T-MEM] resident floor: %.2f GB (%ld bytes)
            [Q3T-MEM] worst peak:     %.2f GB (%ld bytes)
            [Q3T-MEM] transient (peak − floor): %.2f GB (%ld bytes)
            [Q3T-MEM] audio: peak %.3f rms %.3f
            """,
            Self.sentence.count, durSecs, rate,
            loadSecs, genSecs,
            Double(residentFloor) / gib, residentFloor,
            Double(worstPeak) / gib, worstPeak,
            Double(transient) / gib, transient,
            peakAmp, rms))
    }
}
