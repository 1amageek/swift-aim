import MLX
import MLXNN

/// Transformer block (Pre-norm architecture)
///
/// Structure: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
public class TransformerBlock: Module {
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var attn: Attention
    @ModuleInfo public var norm2: LayerNorm
    @ModuleInfo public var mlp: MLP

    /// Initializer
    /// - Parameters:
    ///   - dim: Embedding dimension
    ///   - numHeads: Number of attention heads
    ///   - intermediateSize: Dimension of MLP hidden layer
    ///   - qkvBias: Whether to use bias in QKV projection
    ///   - eps: Epsilon value for LayerNorm
    public init(
        dim: Int,
        numHeads: Int,
        intermediateSize: Int,
        qkvBias: Bool = false,
        eps: Float = 1e-6
    ) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._attn.wrappedValue = Attention(dim: dim, numHeads: numHeads, qkvBias: qkvBias)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._mlp.wrappedValue = MLP(dim: dim, hiddenDim: intermediateSize)
    }

    /// Forward pass
    /// - Parameter x: Input tensor [B, N, D]
    /// - Returns: Output tensor [B, N, D]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm with residual
        var x = x
        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))
        return x
    }
}
