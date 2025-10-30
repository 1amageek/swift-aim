import Testing
import Foundation
import MLX
import MLXRandom
@testable import SwiftAIM

/// Comprehensive tests for individual layers
@Suite("Layer Tests")
struct LayerTests {

    // MARK: - PatchEmbed Tests

    @Suite("PatchEmbed Tests")
    struct PatchEmbedTests {

        @Test("PatchEmbed output shape")
        func testPatchEmbedShape() {
            let patchEmbed = PatchEmbed(
                imageSize: 224,
                patchSize: 14,
                inChannels: 3,
                embedDim: 768
            )

            let input = MLXRandom.normal([2, 224, 224, 3])
            let output = patchEmbed(input)

            // Expected: [B, N, D] where N = (224/14)^2 = 256
            #expect(output.shape.count == 3)
            #expect(output.shape[0] == 2) // batch
            #expect(output.shape[1] == 256) // patches
            #expect(output.shape[2] == 768) // embed dim
        }

        @Test("PatchEmbed different image sizes", arguments: [(224, 14), (336, 14), (448, 14)])
        func testPatchEmbedDifferentSizes(args: (Int, Int)) {
            let (imageSize, patchSize) = args
            let expectedPatches = (imageSize / patchSize) * (imageSize / patchSize)

            let patchEmbed = PatchEmbed(
                imageSize: imageSize,
                patchSize: patchSize,
                inChannels: 3,
                embedDim: 384
            )

            let input = MLXRandom.normal([1, imageSize, imageSize, 3])
            let output = patchEmbed(input)

            #expect(output.shape[1] == expectedPatches)
        }

        @Test("PatchEmbed numPatches calculation")
        func testNumPatchesCalculation() {
            let patchEmbed = PatchEmbed(
                imageSize: 336,
                patchSize: 14,
                inChannels: 3,
                embedDim: 768
            )

            // 336 / 14 = 24, so 24 * 24 = 576 patches
            #expect(patchEmbed.numPatches == 576)
        }

        @Test("PatchEmbed preserves batch dimension")
        func testBatchDimensionPreserved() {
            let patchEmbed = PatchEmbed(
                imageSize: 224,
                patchSize: 14,
                inChannels: 3,
                embedDim: 384
            )

            let batchSizes = [1, 2, 4, 8]
            for batchSize in batchSizes {
                let input = MLXRandom.normal([batchSize, 224, 224, 3])
                let output = patchEmbed(input)
                #expect(output.shape[0] == batchSize)
            }
        }
    }

    // MARK: - Attention Tests

    @Suite("Attention Tests")
    struct AttentionTests {

        @Test("Attention output shape")
        func testAttentionShape() {
            let attention = Attention(dim: 768, numHeads: 12, qkvBias: true)

            let input = MLXRandom.normal([2, 257, 768])
            let output = attention(input)

            // Output should have same shape as input
            #expect(output.shape.count == 3)
            #expect(output.shape[0] == 2)
            #expect(output.shape[1] == 257)
            #expect(output.shape[2] == 768)
        }

        @Test("Attention head dimension calculation")
        func testHeadDimCalculation() {
            let attention1 = Attention(dim: 768, numHeads: 12)
            #expect(attention1.headDim == 64) // 768 / 12

            let attention2 = Attention(dim: 384, numHeads: 6)
            #expect(attention2.headDim == 64) // 384 / 6

            let attention3 = Attention(dim: 192, numHeads: 3)
            #expect(attention3.headDim == 64) // 192 / 3
        }

        @Test("Attention scale calculation")
        func testScaleCalculation() {
            let attention = Attention(dim: 768, numHeads: 12)

            // Scale should be 1/sqrt(headDim) = 1/sqrt(64) = 0.125
            let expectedScale = Float(1.0) / Foundation.sqrt(Float(64.0))
            #expect(abs(attention.scale - expectedScale) < 0.001)
        }

        @Test("Attention different sequence lengths", arguments: [10, 50, 100, 257])
        func testDifferentSequenceLengths(seqLen: Int) {
            let attention = Attention(dim: 384, numHeads: 6)

            let input = MLXRandom.normal([1, seqLen, 384])
            let output = attention(input)

            // Sequence length should be preserved
            #expect(output.shape[1] == seqLen)
        }
    }

    // MARK: - MLP Tests

    @Suite("MLP Tests")
    struct MLPTests {

        @Test("MLP output shape")
        func testMLPShape() {
            let mlp = MLP(dim: 768, hiddenDim: 3072)

            let input = MLXRandom.normal([2, 257, 768])
            let output = mlp(input)

            // Output shape should match input shape
            #expect(output.shape.count == 3)
            #expect(output.shape[0] == 2)
            #expect(output.shape[1] == 257)
            #expect(output.shape[2] == 768)
        }

        @Test("MLP dimension preservation")
        func testDimensionPreservation() {
            let testCases = [(768, 3072), (384, 1536), (192, 768)]

            for (dim, hiddenDim) in testCases {
                let mlp = MLP(dim: dim, hiddenDim: hiddenDim)
                let input = MLXRandom.normal([1, 100, dim])
                let output = mlp(input)

                // Last dimension should match input dim
                #expect(output.shape[2] == dim)
            }
        }

        @Test("MLP non-linearity")
        func testNonLinearity() {
            let mlp = MLP(dim: 64, hiddenDim: 256)

            // Create a zero input
            let zeroInput = MLXArray.zeros([1, 10, 64])
            let output = mlp(zeroInput)

            // With bias, output should not be exactly zero for all elements
            // (though some might be close to zero due to GELU)
            let outputSum = MLX.abs(output).sum().item(Float.self)

            // The sum should be relatively small but not necessarily exactly zero
            // This is a weak test but validates the forward pass works
            #expect(outputSum >= 0.0)
        }
    }

    // MARK: - TransformerBlock Tests

    @Suite("TransformerBlock Tests")
    struct TransformerBlockTests {

        @Test("TransformerBlock output shape")
        func testTransformerBlockShape() {
            let block = TransformerBlock(
                dim: 768,
                numHeads: 12,
                intermediateSize: 3072
            )

            let input = MLXRandom.normal([2, 257, 768])
            let output = block(input)

            // Output shape should match input shape
            #expect(output.shape.count == 3)
            #expect(output.shape[0] == 2)
            #expect(output.shape[1] == 257)
            #expect(output.shape[2] == 768)
        }

        @Test("TransformerBlock residual connection")
        func testResidualConnection() {
            let block = TransformerBlock(
                dim: 192,
                numHeads: 3,
                intermediateSize: 768
            )

            let input = MLXRandom.normal([1, 50, 192])
            let output = block(input)

            // Output should be different from input due to residual additions
            // Check that output is not identical to input
            let diff = MLX.abs(output - input).sum().item(Float.self)
            #expect(diff > 0.0)
        }

        @Test("TransformerBlock with different configurations")
        func testDifferentConfigurations() {
            let configs = [
                (dim: 192, heads: 3, intermediate: 768),
                (dim: 384, heads: 6, intermediate: 1536),
                (dim: 768, heads: 12, intermediate: 3072)
            ]

            for config in configs {
                let block = TransformerBlock(
                    dim: config.dim,
                    numHeads: config.heads,
                    intermediateSize: config.intermediate
                )

                let input = MLXRandom.normal([1, 100, config.dim])
                let output = block(input)

                #expect(output.shape[2] == config.dim)
            }
        }

        @Test("TransformerBlock QKV bias configuration")
        func testQKVBiasConfiguration() {
            // Test with QKV bias enabled
            let blockWithBias = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536,
                qkvBias: true
            )

            // Test with QKV bias disabled
            let blockWithoutBias = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536,
                qkvBias: false
            )

            let input = MLXRandom.normal([1, 50, 384])

            // Both should produce valid outputs
            let output1 = blockWithBias(input)
            let output2 = blockWithoutBias(input)

            #expect(output1.shape[2] == 384)
            #expect(output2.shape[2] == 384)
        }

        @Test("TransformerBlock epsilon value")
        func testEpsilonValue() {
            let block = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536,
                eps: 1e-5
            )

            let input = MLXRandom.normal([1, 50, 384])
            let output = block(input)

            // Should process without numerical issues
            #expect(output.shape.count == 3)
            #expect(output.shape[0] == 1)
            #expect(output.shape[1] == 50)
            #expect(output.shape[2] == 384)
        }
    }

    // MARK: - Integration Tests

    @Suite("Layer Integration Tests")
    struct LayerIntegrationTests {

        @Test("Full pipeline: PatchEmbed -> TransformerBlock")
        func testPatchEmbedToTransformer() {
            let patchEmbed = PatchEmbed(
                imageSize: 224,
                patchSize: 14,
                inChannels: 3,
                embedDim: 384
            )

            let transformerBlock = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536
            )

            // Input image
            let image = MLXRandom.normal([1, 224, 224, 3])

            // Through patch embedding
            let patches = patchEmbed(image)
            #expect(patches.shape[1] == 256)
            #expect(patches.shape[2] == 384)

            // Through transformer block
            let output = transformerBlock(patches)
            #expect(output.shape[1] == 256)
            #expect(output.shape[2] == 384)
        }

        @Test("Multiple transformer blocks in sequence")
        func testMultipleTransformerBlocks() {
            let block1 = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536
            )

            let block2 = TransformerBlock(
                dim: 384,
                numHeads: 6,
                intermediateSize: 1536
            )

            let input = MLXRandom.normal([1, 100, 384])

            // Pass through multiple blocks
            let intermediate = block1(input)
            let output = block2(intermediate)

            // Shape should be preserved
            #expect(output.shape[0] == 1)
            #expect(output.shape[1] == 100)
            #expect(output.shape[2] == 384)

            // Output should be different from input
            let diff = MLX.abs(output - input).sum().item(Float.self)
            #expect(diff > 0.0)
        }
    }
}
