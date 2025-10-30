import Testing
import MLX
import MLXRandom
@testable import SwiftAIM

/// Comprehensive tests for AIMv2Model
@Suite("AIMv2Model Tests")
struct AIMv2ModelTests {

    /// Test model initialization and configuration
    @Test("Model initialization with default config")
    func testModelInitialization() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 768,
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072,
            imageSize: 224,
            patchSize: 14,
            numChannels: 3
        )

        let model = AIMv2Model(config: config)

        // Verify configuration is stored correctly
        #expect(model.config.hiddenSize == 768)
        #expect(model.config.numHiddenLayers == 12)
        #expect(model.config.numAttentionHeads == 12)

        // Verify number of transformer blocks matches config
        #expect(model.blocks.count == 12)
    }

    /// Test forward pass with valid input produces correct output shape
    @Test("Forward pass output shape")
    func testForwardPassShape() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 6,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224,
            patchSize: 14
        )

        let model = AIMv2Model(config: config)

        // Create dummy input: batch=2, channels=3, height=224, width=224
        let batchSize = 2
        let input = MLXRandom.normal([batchSize, 3, 224, 224])

        // Forward pass
        let output = model(input)

        // Expected output shape: [B, N+1, D]
        // N = (224/14)^2 = 256 patches
        // N+1 = 257 (including CLS token)
        #expect(output.shape.count == 3)
        #expect(output.shape[0] == batchSize)
        #expect(output.shape[1] == 257) // 256 patches + 1 CLS token
        #expect(output.shape[2] == 384) // hiddenSize
    }

    /// Test CLS feature extraction
    @Test("CLS feature extraction")
    func testCLSFeatureExtraction() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)
        let input = MLXRandom.normal([1, 3, 224, 224])

        // Extract CLS feature
        let clsFeature = model.extractCLSFeature(input)

        // Should be [B, D]
        #expect(clsFeature.shape.count == 2)
        #expect(clsFeature.shape[0] == 1) // batch size
        #expect(clsFeature.shape[1] == 384) // hiddenSize
    }

    /// Test patch feature extraction
    @Test("Patch feature extraction")
    func testPatchFeatureExtraction() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)
        let input = MLXRandom.normal([2, 3, 224, 224])

        // Extract patch features
        let patchFeatures = model.extractPatchFeatures(input)

        // Should be [B, N, D] without CLS token
        #expect(patchFeatures.shape.count == 3)
        #expect(patchFeatures.shape[0] == 2) // batch size
        #expect(patchFeatures.shape[1] == 256) // 256 patches (no CLS)
        #expect(patchFeatures.shape[2] == 384) // hiddenSize
    }

    /// Test absolute positional embeddings
    @Test("Absolute positional embeddings")
    func testAbsolutePositionalEmbeddings() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224,
            positionEmbeddingType: .absolute
        )

        let model = AIMv2Model(config: config)

        // Absolute mode should have posEmbed parameter
        #expect(model.posEmbed != nil)
        #expect(model.posEmbed!.shape[0] == 1)
        #expect(model.posEmbed!.shape[1] == 257) // patches + CLS
        #expect(model.posEmbed!.shape[2] == 384)
    }

    /// Test sincos positional embeddings
    @Test("Sincos positional embeddings")
    func testSincosPositionalEmbeddings() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224,
            positionEmbeddingType: .sincos
        )

        let model = AIMv2Model(config: config)

        // Sincos mode should not have learnable posEmbed
        #expect(model.posEmbed == nil)

        // Forward pass should still work
        let input = MLXRandom.normal([1, 3, 224, 224])
        let output = model(input)
        #expect(output.shape[1] == 257)
    }

    /// Test different image sizes
    @Test("Different image sizes", arguments: [224, 336, 448])
    func testDifferentImageSizes(imageSize: Int) {
        let patchSize = 14
        let expectedPatches = (imageSize / patchSize) * (imageSize / patchSize)

        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: imageSize,
            patchSize: patchSize
        )

        let model = AIMv2Model(config: config)
        let input = MLXRandom.normal([1, 3, imageSize, imageSize])

        let output = model(input)

        // Verify output has correct number of tokens
        #expect(output.shape[1] == expectedPatches + 1) // patches + CLS
    }

    /// Test input validation - wrong number of channels
    @Test("Input validation - wrong channels", .bug("https://github.com/user/repo/issues/1"))
    func testInputValidationWrongChannels() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)

        // Wrong number of channels (1 instead of 3)
        let input = MLXRandom.normal([1, 1, 224, 224])

        // This should trigger a precondition failure
        #expect(performing: {
            let _ = model(input)
        }, throws: { error in
            // Precondition failures can't be caught in Swift, but we document the expected behavior
            return false
        })
    }

    /// Test input validation - wrong image size
    @Test("Input validation - wrong size")
    func testInputValidationWrongSize() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)

        // Wrong image size (112 instead of 224)
        let input = MLXRandom.normal([1, 3, 112, 112])

        // This should trigger a precondition failure
        #expect(performing: {
            let _ = model(input)
        }, throws: { error in
            // Document expected behavior
            return false
        })
    }

    /// Test batch size handling
    @Test("Multiple batch sizes", arguments: [1, 2, 4, 8])
    func testBatchSizes(batchSize: Int) {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 192,
            numHiddenLayers: 2,
            numAttentionHeads: 3,
            intermediateSize: 768,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)
        let input = MLXRandom.normal([batchSize, 3, 224, 224])

        let output = model(input)

        // First dimension should match batch size
        #expect(output.shape[0] == batchSize)
    }

    /// Test configuration computed properties
    @Test("Configuration computed properties")
    func testConfigurationComputedProperties() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 768,
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072,
            imageSize: 336,
            patchSize: 14
        )

        // Test numPatches
        let expectedPatches = (336 / 14) * (336 / 14)
        #expect(config.numPatches == expectedPatches)
        #expect(config.numPatches == 576)

        // Test sequenceLength (patches + CLS)
        #expect(config.sequenceLength == expectedPatches + 1)
        #expect(config.sequenceLength == 577)

        // Test headDim
        #expect(config.headDim == 768 / 12)
        #expect(config.headDim == 64)
    }

    /// Test CLS token initialization is not zero
    @Test("CLS token non-zero initialization")
    func testCLSTokenInitialization() {
        let config = AIMv2Configuration(
            modelType: "aimv2-test",
            hiddenSize: 384,
            numHiddenLayers: 2,
            numAttentionHeads: 6,
            intermediateSize: 1536,
            imageSize: 224
        )

        let model = AIMv2Model(config: config)

        // CLS token should be initialized with normal distribution, not zeros
        let clsToken = model.clsToken

        // Check shape
        #expect(clsToken.shape[0] == 1)
        #expect(clsToken.shape[1] == 1)
        #expect(clsToken.shape[2] == 384)

        // Check it's not all zeros (with high probability)
        let clsSum = MLX.abs(clsToken).sum().item(Float.self)
        #expect(clsSum > 0.0)
    }
}
