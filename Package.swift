// swift-tools-version: 6.2
import PackageDescription

// mlx-qwen3-tts-swift — a `tts` surface over Qwen3-TTS (Talker + CodePredictor + Mimi-based
// SpeechTokenizer codec, 24 kHz). Backed by our qwen3-tts-mlx-swift core (Apache-2.0, derived
// from soniqo/speech-swift) consuming mlx-community checkpoints — Base / CustomVoice /
// VoiceDesign × 0.6B / 1.7B × {4,5,6,8-bit, bf16}. Includes ICL voice cloning (~0.89 speaker
// similarity) via the canonical `VoiceSelector.referenceAudio` path.
//
// Swift-port naming: `-swift` on the package/repo; module/product stays clean `MLXQwen3TTS`.
let package = Package(
    name: "mlx-qwen3-tts-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXQwen3TTS", targets: ["MLXQwen3TTS"]),
    ],
    dependencies: [
        // The engine contract is consumed via tagged URL; the Qwen3TTS core is pinned to a tagged
        // release. Bumped to 0.27.0 for the CAN cancellation-conformance gate
        // (MLXServeConformance.CancellationConformance). No swift-huggingface dep needed here —
        // downloads ride the core's HuggingFaceDownloader, which already lands snapshots in the
        // store layout.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.27.0"),
        .package(url: "https://github.com/xocialize/qwen3-tts-mlx-swift.git", from: "0.1.0"),
        // mlx-swift, for MLX.Memory.clearCache() in the wrapper's unload(). Pinned to the
        // revision the Qwen3TTS core already resolves (0.31.4) so this adds no resolution churn.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
    ],
    targets: [
        .target(
            name: "MLXQwen3TTS",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Qwen3TTS", package: "qwen3-tts-mlx-swift"),
                .product(name: "AudioCommon", package: "qwen3-tts-mlx-swift"),
                .product(name: "Qwen3TTSCloning", package: "qwen3-tts-mlx-swift"),
            ],
            // The core builds in Swift 5 language mode (upstream-derived code, non-Sendable model
            // types). The engine serializes all lifecycle calls on InferenceActor (no real
            // concurrency), so v5 mode keeps strict region-isolation sends a warning rather than
            // a hard error — same precedent as mlx-kokoro-tts-swift.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXQwen3TTSTests",
            dependencies: [
                "MLXQwen3TTS",
                // Test-only: admissibility sanity check through the engine.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                // The offline MAT-1..5 materialization gate.
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
