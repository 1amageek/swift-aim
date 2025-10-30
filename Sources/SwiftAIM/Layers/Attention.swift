import MLX
import MLXNN
import MLXFast
import Foundation

/// Multi-head self-attention mechanism
///
/// Splits Query, Key, and Value into multiple heads and performs attention computation in parallel.
public class Attention: Module {
    @ModuleInfo public var qkv: Linear
    @ModuleInfo public var proj: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    /// Initializer
    /// - Parameters:
    ///   - dim: Embedding dimension
    ///   - numHeads: Number of attention heads
    ///   - qkvBias: Whether to use bias in QKV projection
    public init(dim: Int, numHeads: Int = 8, qkvBias: Bool = false) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads

        // スケール計算: 1 / sqrt(headDim)
        self.scale = pow(Float(self.headDim), -0.5)

        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim)
    }

    /// Forward pass
    /// - Parameters:
    ///   - x: Input tensor [B, N, D]
    ///   - mask: Optional attention mask
    /// - Returns: Tensor after attention application [B, N, D]
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]
        let N = x.shape[1]
        let inputDim = x.shape[2]
        let D = numHeads * headDim

        // 入力次元の検証（リリースビルドでも有効）
        precondition(inputDim == D,
                     "Input dimension (\(inputDim)) doesn't match expected (\(D) = \(numHeads) * \(headDim))")

        // QKV projection: [B, N, D] -> [B, N, 3*D]
        let qkv = self.qkv(x)
            .reshaped(B, N, 3, numHeads, headDim)
            .transposed(2, 0, 3, 1, 4)

        let q = qkv[0]  // [B, numHeads, N, headDim]
        let k = qkv[1]
        let v = qkv[2]

        // Scaled dot-product attention
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        // [B, numHeads, N, headDim] -> [B, N, D]
        let out = attn
            .transposed(0, 2, 1, 3)
            .reshaped(B, N, D)

        return proj(out)
    }
}
