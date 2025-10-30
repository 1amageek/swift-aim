import MLX
import MLXNN
import MLXRandom
import Foundation

/// AIMv2 Vision Encoder (inference only)
///
/// Executes pre-trained AIMv2 models from HuggingFace.
public class AIMv2Model: Module {
    @ModuleInfo public var patchEmbed: PatchEmbed
    @ModuleInfo public var blocks: [TransformerBlock]
    @ModuleInfo public var norm: LayerNorm

    public let config: AIMv2Configuration
    @ParameterInfo public var clsToken: MLXArray
    @ParameterInfo public var posEmbed: MLXArray?

    /// Cache for sincos positional embeddings (used only in .sincos mode)
    private var sinCosCache: MLXArray?

    /// Initializer
    /// - Parameter config: Model configuration
    public init(config: AIMv2Configuration) {
        self.config = config

        // CLS token (learnable parameter)
        // Standard initialization method for Transformer/ViT (normal distribution, std=0.02)
        self._clsToken.wrappedValue = MLXRandom.normal(
            [1, 1, config.hiddenSize],
            loc: 0.0,
            scale: 0.02
        )

        // Patch embedding
        self._patchEmbed.wrappedValue = PatchEmbed(
            imageSize: config.imageSize,
            patchSize: config.patchSize,
            inChannels: config.numChannels,
            embedDim: config.hiddenSize
        )

        // Positional embedding (learnable parameter only for absolute mode)
        if config.positionEmbeddingType == .absolute {
            self._posEmbed.wrappedValue = MLXRandom.normal([1, config.sequenceLength, config.hiddenSize])
        } else {
            self._posEmbed.wrappedValue = nil
        }

        // Transformer blocks
        self._blocks.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            TransformerBlock(
                dim: config.hiddenSize,
                numHeads: config.numAttentionHeads,
                intermediateSize: config.intermediateSize,
                qkvBias: config.qkvBias,
                eps: config.layerNormEps
            )
        }

        // Final normalization
        self._norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        // Call super.init() to complete initialization
        super.init()

        // Precompute sincos positional embeddings if needed (after super.init)
        if config.positionEmbeddingType == .sincos {
            self.sinCosCache = createSinCosPositionalEmbedding(
                numPatches: config.numPatches,
                embedDim: config.hiddenSize
            )
        }
    }

    /// Forward pass (inference)
    /// - Parameter pixels: Image tensor [B, C, H, W]
    /// - Returns: Feature tensor [B, N+1, D] (N=number of patches, D=hiddenSize)
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        // Input validation
        precondition(pixels.ndim == 4, "Expected 4D input [B, C, H, W], got \(pixels.ndim)D")
        precondition(pixels.shape[1] == config.numChannels,
                     "Expected \(config.numChannels) channels, got \(pixels.shape[1])")
        precondition(pixels.shape[2] == config.imageSize && pixels.shape[3] == config.imageSize,
                     "Expected \(config.imageSize)x\(config.imageSize) images, got \(pixels.shape[2])x\(pixels.shape[3])")

        // Patch embedding: [B, C, H, W] -> [B, N, D]
        var x = patchEmbed(pixels)

        // Add CLS token: [B, N, D] -> [B, N+1, D]
        let B = x.shape[0]
        let clsTokens = MLX.broadcast(clsToken, to: [B, 1, config.hiddenSize])
        x = MLX.concatenated([clsTokens, x], axis: 1)

        // Add positional embeddings
        if let posEmbed = posEmbed ?? sinCosCache {
            x = x + posEmbed
        }

        // Pass through Transformer blocks
        for block in blocks {
            x = block(x)
        }

        // Final normalization
        x = norm(x)

        return x
    }

    /// Extract CLS feature (for classification tasks)
    /// - Parameter pixels: Image tensor [B, C, H, W]
    /// - Returns: CLS feature [B, D]
    public func extractCLSFeature(_ pixels: MLXArray) -> MLXArray {
        let features = self(pixels)
        return features[0..., 0, 0...]  // First token (CLS)
    }

    /// Extract all patch features (for detection tasks)
    /// - Parameter pixels: Image tensor [B, C, H, W]
    /// - Returns: Patch features [B, N, D]
    public func extractPatchFeatures(_ pixels: MLXArray) -> MLXArray {
        let features = self(pixels)
        return features[0..., 1..., 0...]  // Patch tokens excluding CLS
    }

    /// Generate sincos positional embeddings (for 2D grid)
    private func createSinCosPositionalEmbedding(
        numPatches: Int,
        embedDim: Int
    ) -> MLXArray {
        // Dimension validation
        precondition(embedDim % 4 == 0,
                     "embedDim must be divisible by 4 for sincos positional embedding, got \(embedDim)")
        precondition(numPatches > 0,
                     "numPatches must be positive, got \(numPatches)")

        let gridSize = Int(pow(Double(numPatches), 0.5))
        precondition(gridSize * gridSize == numPatches,
                     "numPatches must be a perfect square, got \(numPatches)")

        let embedDimQuarter = embedDim / 4

        // Calculate frequencies
        var omega = MLXArray(0..<embedDimQuarter).asType(.float32) / Float(embedDimQuarter)
        omega = 1.0 / pow(10000.0, omega)

        // Sin/cos embeddings for Y and X coordinates
        var posEmbed: [MLXArray] = []

        for y in 0..<gridSize {
            for x in 0..<gridSize {
                let yPos = Float(y)
                let xPos = Float(x)

                // Sin/cos for Y coordinate
                let yEmbed = yPos * omega
                let ySin = sin(yEmbed)
                let yCos = cos(yEmbed)

                // Sin/cos for X coordinate
                let xEmbed = xPos * omega
                let xSin = sin(xEmbed)
                let xCos = cos(xEmbed)

                // Create [embedDim] vector
                let embed = MLX.concatenated([ySin, yCos, xSin, xCos], axis: 0)
                posEmbed.append(embed)
            }
        }

        // [numPatches, embedDim]
        let patchEmbed = MLX.stacked(posEmbed, axis: 0)

        // Add zero vector for CLS token
        let clsEmbed = MLXArray.zeros([1, embedDim])

        // [1, numPatches + 1, embedDim]
        return MLX.concatenated([clsEmbed, patchEmbed], axis: 0).expandedDimensions(axis: 0)
    }
}

// MARK: - Weight Sanitization

extension AIMv2Model {
    /// Convert PyTorch weights to MLX format
    /// - Parameter weights: Weight dictionary in PyTorch format
    /// - Returns: Weight dictionary in MLX format
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        var skippedKeys: [String] = []

        for (key, value) in weights {
            // Size validation
            guard value.ndim <= 4 else {
                print("⚠️  Warning: Skipping '\(key)' with unusual ndim: \(value.ndim)")
                skippedKeys.append(key)
                continue
            }

            let totalElements = value.shape.reduce(1, *)
            guard totalElements < 100_000_000 else {  // 100M要素
                print("⚠️  Warning: Skipping '\(key)' with excessive size: \(totalElements) elements")
                skippedKeys.append(key)
                continue
            }

            var newKey = key
            var newValue = value

            // Remove prefixes
            newKey = newKey.replacingOccurrences(of: "vision_model.", with: "")
            newKey = newKey.replacingOccurrences(of: "encoder.", with: "")

            // Transpose Conv2d weights (strict key matching)
            // PyTorch: [out, in, kH, kW] -> MLX: [out, kH, kW, in]
            if newKey == "patch_embed.projection.weight" ||
               newKey == "patchEmbed.projection.weight" {
                guard value.ndim == 4 else {
                    print("⚠️  Warning: Conv2d weight '\(key)' has unexpected ndim: \(value.ndim), expected 4")
                    skippedKeys.append(key)
                    continue
                }
                newValue = value.transposed(0, 2, 3, 1)
            }

            sanitized[newKey] = newValue
        }

        if !skippedKeys.isEmpty {
            print("ℹ️  Skipped \(skippedKeys.count) keys during weight sanitization")
        }

        return sanitized
    }
}
