import MLX
import MLXNN

/// Layer that converts images to patch embeddings
///
/// Divides input images into 14x14 patches and converts them to embedding vectors using Conv2d.
public class PatchEmbed: Module, UnaryLayer {
    @ModuleInfo public var projection: Conv2d

    public let imageSize: Int
    public let patchSize: Int
    public let numPatches: Int

    /// Initializer
    /// - Parameters:
    ///   - imageSize: Input image size (224, 336, 448, etc.)
    ///   - patchSize: Patch size (fixed at 14)
    ///   - inChannels: Number of input channels (RGB=3)
    ///   - embedDim: Embedding dimension (768, etc.)
    public init(
        imageSize: Int = 224,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 768
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)

        self._projection.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize),
            padding: IntOrPair(0)
        )
    }

    /// Forward pass
    /// - Parameter x: Input image [B, C, H, W]
    /// - Returns: Patch embeddings [B, N, D]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // [B, C, H, W] -> [B, embedDim, H', W']
        var x = projection(x)

        let B = x.shape[0]
        let D = x.shape[1]

        // Shape validation: verify patch count is as expected
        let H_prime = x.shape[2]
        let W_prime = x.shape[3]
        let actualPatches = H_prime * W_prime
        precondition(actualPatches == numPatches,
                     "Patch count mismatch: expected \(numPatches) patches, got \(actualPatches)")

        // [B, D, H', W'] -> [B, D, N]
        x = x.reshaped(B, D, numPatches)

        // [B, D, N] -> [B, N, D]
        return x.transposed(0, 2, 1)
    }
}
