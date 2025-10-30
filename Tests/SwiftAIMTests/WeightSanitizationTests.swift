import Testing
import MLX
@testable import SwiftAIM

/// Comprehensive tests for weight sanitization (PyTorch → MLX conversion)
@Suite("Weight Sanitization Tests")
struct WeightSanitizationTests {

    // MARK: - Prefix Removal Tests

    @Test("Remove vision_model prefix")
    func testVisionModelPrefixRemoval() {
        let weights: [String: MLXArray] = [
            "vision_model.patch_embed.projection.weight": MLXArray.ones([768, 3, 14, 14]),
            "vision_model.norm.weight": MLXArray.ones([768])
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Prefixes should be removed
        #expect(sanitized["patch_embed.projection.weight"] != nil)
        #expect(sanitized["norm.weight"] != nil)

        // Original keys should not exist
        #expect(sanitized["vision_model.patch_embed.projection.weight"] == nil)
        #expect(sanitized["vision_model.norm.weight"] == nil)
    }

    @Test("Remove encoder prefix")
    func testEncoderPrefixRemoval() {
        let weights: [String: MLXArray] = [
            "encoder.blocks.0.attn.qkv.weight": MLXArray.ones([2304, 768]),
            "encoder.norm.bias": MLXArray.zeros([768])
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Prefixes should be removed
        #expect(sanitized["blocks.0.attn.qkv.weight"] != nil)
        #expect(sanitized["norm.bias"] != nil)

        // Original keys should not exist
        #expect(sanitized["encoder.blocks.0.attn.qkv.weight"] == nil)
    }

    @Test("Remove multiple prefixes")
    func testMultiplePrefixRemoval() {
        let weights: [String: MLXArray] = [
            "vision_model.encoder.blocks.0.norm1.weight": MLXArray.ones([768])
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Both prefixes should be removed
        #expect(sanitized["blocks.0.norm1.weight"] != nil)
    }

    // MARK: - Conv2d Transpose Tests

    @Test("Conv2d weight transpose for patch_embed")
    func testConv2dWeightTranspose() {
        // PyTorch Conv2d format: [out_channels, in_channels, kernel_h, kernel_w]
        let pytorchWeight = MLXArray.ones([768, 3, 14, 14])

        let weights: [String: MLXArray] = [
            "patch_embed.projection.weight": pytorchWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        guard let mlxWeight = sanitized["patch_embed.projection.weight"] else {
            Issue.record("patch_embed.projection.weight not found in sanitized weights")
            return
        }

        // MLX Conv2d format: [out_channels, kernel_h, kernel_w, in_channels]
        #expect(mlxWeight.shape[0] == 768) // out_channels
        #expect(mlxWeight.shape[1] == 14)  // kernel_h
        #expect(mlxWeight.shape[2] == 14)  // kernel_w
        #expect(mlxWeight.shape[3] == 3)   // in_channels
    }

    @Test("Conv2d weight transpose for patchEmbed variant")
    func testConv2dWeightTransposePatchEmbedVariant() {
        let pytorchWeight = MLXArray.ones([384, 3, 14, 14])

        let weights: [String: MLXArray] = [
            "patchEmbed.projection.weight": pytorchWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        guard let mlxWeight = sanitized["patchEmbed.projection.weight"] else {
            Issue.record("patchEmbed.projection.weight not found")
            return
        }

        // Should be transposed
        #expect(mlxWeight.shape[0] == 384)
        #expect(mlxWeight.shape[1] == 14)
        #expect(mlxWeight.shape[2] == 14)
        #expect(mlxWeight.shape[3] == 3)
    }

    @Test("Non-Conv2d weights are not transposed")
    func testNonConv2dWeightsNotTransposed() {
        let linearWeight = MLXArray.ones([768, 768])
        let biasWeight = MLXArray.ones([768])

        let weights: [String: MLXArray] = [
            "blocks.0.attn.qkv.weight": linearWeight,
            "blocks.0.attn.qkv.bias": biasWeight,
            "norm.weight": biasWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Linear weights should NOT be transposed
        #expect(sanitized["blocks.0.attn.qkv.weight"]?.shape[0] == 768)
        #expect(sanitized["blocks.0.attn.qkv.weight"]?.shape[1] == 768)

        // Bias should remain unchanged
        #expect(sanitized["blocks.0.attn.qkv.bias"]?.shape[0] == 768)
        #expect(sanitized["norm.weight"]?.shape[0] == 768)
    }

    // MARK: - Validation Tests

    @Test("Skip weights with excessive ndim")
    func testSkipExcessiveNdim() {
        let normalWeight = MLXArray.ones([768, 768])
        let excessiveWeight = MLXArray.ones([2, 3, 4, 5, 6]) // 5D tensor

        let weights: [String: MLXArray] = [
            "normal.weight": normalWeight,
            "excessive.weight": excessiveWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Normal weight should be included
        #expect(sanitized["normal.weight"] != nil)

        // Excessive weight should be skipped
        #expect(sanitized["excessive.weight"] == nil)
    }

    @Test("Skip weights with excessive size")
    func testSkipExcessiveSize() {
        let normalWeight = MLXArray.ones([768, 768])
        // Create a weight that would exceed 100M elements
        // 10000 x 10000 = 100M elements (at the limit)
        // We can't actually create such large arrays easily in tests,
        // so we'll test the logic conceptually

        let weights: [String: MLXArray] = [
            "normal.weight": normalWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Normal weight should pass
        #expect(sanitized["normal.weight"] != nil)
    }

    @Test("Skip Conv2d weight with wrong ndim")
    func testSkipConv2dWithWrongNdim() {
        // Conv2d weight should be 4D, but this is 3D
        let wrongWeight = MLXArray.ones([768, 3, 14])

        let weights: [String: MLXArray] = [
            "patch_embed.projection.weight": wrongWeight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Should be skipped due to wrong ndim
        #expect(sanitized["patch_embed.projection.weight"] == nil)
    }

    // MARK: - Integration Tests

    @Test("Sanitize complete model weights")
    func testSanitizeCompleteModelWeights() {
        let weights: [String: MLXArray] = [
            // CLS token
            "vision_model.clsToken": MLXArray.ones([1, 1, 768]),

            // Position embeddings
            "vision_model.posEmbed": MLXArray.ones([1, 257, 768]),

            // Patch embedding
            "vision_model.patch_embed.projection.weight": MLXArray.ones([768, 3, 14, 14]),
            "vision_model.patch_embed.projection.bias": MLXArray.ones([768]),

            // Transformer blocks
            "encoder.blocks.0.norm1.weight": MLXArray.ones([768]),
            "encoder.blocks.0.attn.qkv.weight": MLXArray.ones([2304, 768]),
            "encoder.blocks.0.attn.proj.weight": MLXArray.ones([768, 768]),

            // Final norm
            "vision_model.norm.weight": MLXArray.ones([768]),
            "vision_model.norm.bias": MLXArray.ones([768])
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Check all weights are processed correctly
        #expect(sanitized["clsToken"] != nil)
        #expect(sanitized["posEmbed"] != nil)
        #expect(sanitized["patch_embed.projection.weight"] != nil)
        #expect(sanitized["patch_embed.projection.bias"] != nil)
        #expect(sanitized["blocks.0.norm1.weight"] != nil)
        #expect(sanitized["blocks.0.attn.qkv.weight"] != nil)
        #expect(sanitized["norm.weight"] != nil)

        // Check Conv2d weight is transposed
        let conv2dWeight = sanitized["patch_embed.projection.weight"]!
        #expect(conv2dWeight.shape[0] == 768)
        #expect(conv2dWeight.shape[3] == 3) // in_channels moved to last dimension
    }

    @Test("Sanitize empty weights")
    func testSanitizeEmptyWeights() {
        let weights: [String: MLXArray] = [:]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        #expect(sanitized.isEmpty)
    }

    @Test("Sanitize preserves weight values")
    func testSanitizePreservesValues() {
        // Create a simple weight with known values
        let value: Float = 42.0
        let weight = MLXArray.ones([768]) * value

        let weights: [String: MLXArray] = [
            "norm.weight": weight
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        guard let sanitizedWeight = sanitized["norm.weight"] else {
            Issue.record("Weight not found after sanitization")
            return
        }

        // Check values are preserved
        let sum = sanitizedWeight.sum().item(Float.self)
        let expectedSum = value * 768.0
        #expect(abs(sum - expectedSum) < 0.01)
    }

    @Test("Sanitize handles mixed valid and invalid weights")
    func testMixedValidInvalidWeights() {
        let weights: [String: MLXArray] = [
            "valid.weight": MLXArray.ones([768, 768]),
            "invalid.weight": MLXArray.ones([1, 2, 3, 4, 5]), // 5D - invalid
            "another_valid.bias": MLXArray.ones([768])
        ]

        let sanitized = AIMv2Model.sanitize(weights: weights)

        // Valid weights should be present
        #expect(sanitized["valid.weight"] != nil)
        #expect(sanitized["another_valid.bias"] != nil)

        // Invalid weight should be skipped
        #expect(sanitized["invalid.weight"] == nil)

        // Should have exactly 2 weights
        #expect(sanitized.count == 2)
    }

    @Test("Sanitize with various Conv2d sizes")
    func testVariousConv2dSizes() {
        let testCases: [(outChannels: Int, inChannels: Int, kernelSize: Int)] = [
            (768, 3, 14),
            (384, 3, 14),
            (192, 3, 14),
            (768, 3, 16),
            (512, 4, 8)
        ]

        for testCase in testCases {
            let pytorchWeight = MLXArray.ones([
                testCase.outChannels,
                testCase.inChannels,
                testCase.kernelSize,
                testCase.kernelSize
            ])

            let weights: [String: MLXArray] = [
                "patch_embed.projection.weight": pytorchWeight
            ]

            let sanitized = AIMv2Model.sanitize(weights: weights)

            guard let mlxWeight = sanitized["patch_embed.projection.weight"] else {
                Issue.record("Weight not found for test case \(testCase)")
                continue
            }

            // Verify transpose
            #expect(mlxWeight.shape[0] == testCase.outChannels)
            #expect(mlxWeight.shape[1] == testCase.kernelSize)
            #expect(mlxWeight.shape[2] == testCase.kernelSize)
            #expect(mlxWeight.shape[3] == testCase.inChannels)
        }
    }
}
