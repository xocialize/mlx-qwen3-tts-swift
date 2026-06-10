import Foundation
import Testing
import MLXToolKit
@testable import MLXQwen3TTS

// Offline conformance tests — no MLX kernels, no weights (the metallib boundary).
// Live inference is proven in the MLXEngine Testing app (see APP-VALIDATION.md).

@Suite struct CheckpointCatalogTests {
    @Test func defaultCheckpointComputesExpectedRepoID() {
        #expect(Qwen3TTSCheckpoint.default.repoID == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
    }

    @Test func repoIDsMatchCommunityNaming() {
        let cv = Qwen3TTSCheckpoint(variant: .customVoice, size: .s0_6B, quant: .q4)
        #expect(cv.repoID == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
        let vd = Qwen3TTSCheckpoint(variant: .voiceDesign, size: .s1_7B, quant: .bf16)
        #expect(vd.repoID == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    }

    @Test func catalogExcludesUnpublishedVoiceDesign0_6B() {
        // 3 variants × 2 sizes × 5 quants = 30, minus VoiceDesign×0.6B×5 = 25.
        #expect(Qwen3TTSCheckpoint.allPublished.count == 25)
        #expect(!Qwen3TTSCheckpoint.allPublished.contains {
            $0.variant == .voiceDesign && $0.size == .s0_6B
        })
    }

    @Test func footprintsGrowWithQuantWidth() {
        let q4 = Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .q4)
        let q8 = Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .q8)
        let bf16 = Qwen3TTSCheckpoint(variant: .base, size: .s1_7B, quant: .bf16)
        #expect(q4.estimatedResidentBytes < q8.estimatedResidentBytes)
        #expect(q8.estimatedResidentBytes < bf16.estimatedResidentBytes)
    }
}

@Suite struct ConfigurationTests {
    @Test func codableRoundTripPreservesCheckpoint() throws {
        var config = Qwen3TTSConfiguration(
            checkpoint: Qwen3TTSCheckpoint(variant: .customVoice, size: .s1_7B, quant: .q6),
            defaultLanguage: "japanese")
        config.modelsRootDirectory = URL(fileURLWithPath: "/tmp/models")

        let decoded = try JSONDecoder().decode(
            Qwen3TTSConfiguration.self, from: JSONEncoder().encode(config))
        #expect(decoded.checkpoint == config.checkpoint)
        #expect(decoded.defaultLanguage == "japanese")
        // Environment-specific, deliberately not portable config.
        #expect(decoded.modelsRootDirectory == nil)
    }

    @Test func defaultConfigurationIsDefaultable() {
        let config = Qwen3TTSConfiguration()
        #expect(config.checkpoint == .default)
        #expect(config.defaultLanguage == "english")
        #expect(config.modelsRootDirectory == nil)
    }
}

@Suite struct ManifestTests {
    @Test func manifestRegistersOneTTSSurface() {
        let manifest = Qwen3TTSPackage.manifest
        #expect(manifest.capabilities == [.tts])
        #expect(manifest.surfaces.count == 1)
        #expect(manifest.surfaces[0].name == "qwen3-tts")
    }

    @Test func licensePassesPermissiveGateOnBothLayers() {
        let result = LicensePolicy.permissiveOnly.evaluate(Qwen3TTSPackage.manifest.license)
        #expect(result == .admitted)
    }

    @Test func requirementsDeclareMetalAndFootprints() {
        let requirements = Qwen3TTSPackage.manifest.requirements
        #expect(requirements.requiredBackends.contains(.metalGPU))
        #expect(!requirements.footprints.isEmpty)
        for footprint in requirements.footprints {
            #expect(footprint.residentBytes > 1_000_000_000) // codec stack + headroom floor
        }
    }

    @Test func registrationFactoryBuildsFromTypedConfiguration() throws {
        let registration = Qwen3TTSPackage.registration
        let package = try registration.makePackage(Qwen3TTSConfiguration())
        #expect(package is Qwen3TTSPackage)
    }

    @Test func registrationFactoryRejectsForeignConfiguration() {
        let registration = Qwen3TTSPackage.registration
        #expect(throws: PackageError.self) {
            _ = try registration.makePackage(
                StandardConfiguration(weightsRepo: "other/repo"))
        }
    }
}

@Suite struct WAVCodecTests {
    @Test func encodeWAV16ProducesValidHeader() {
        let samples: [Float] = [0, 0.5, -0.5, 1.0, -1.0]
        let data = Qwen3TTSPackage.encodeWAV16(samples: samples, sampleRate: 24_000)

        #expect(data.count == 44 + samples.count * 2)
        #expect(String(data: data.prefix(4), encoding: .ascii) == "RIFF")
        #expect(String(data: data.subdata(in: 8..<12), encoding: .ascii) == "WAVE")
        // 24 kHz mono 16-bit at offset 24 (sample rate, little-endian)
        let rate = data.subdata(in: 24..<28).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(UInt32(littleEndian: rate) == 24_000)
    }

    @Test func wavRoundTripPreservesSamples() throws {
        let samples: [Float] = (0..<480).map { sin(Float($0) * 0.1) * 0.8 }
        let wav = Qwen3TTSPackage.encodeWAV16(samples: samples, sampleRate: 24_000)
        let audio = Audio(format: .wav, data: wav, sampleRate: 24_000, channels: 1)

        let (decoded, rate) = try Qwen3TTSPackage.decodeWAV(audio)
        #expect(rate == 24_000)
        #expect(decoded.count == samples.count)
        for (a, b) in zip(decoded, samples) {
            #expect(abs(a - b) < 0.001) // 16-bit quantization tolerance
        }
    }
}
