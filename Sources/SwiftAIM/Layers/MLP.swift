import MLX
import MLXNN

/// Feed-forward network (Multi-Layer Perceptron)
///
/// Consists of two linear layers with a GELU activation function.
public class MLP: Module {
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear

    /// Initializer
    /// - Parameters:
    ///   - dim: Input/output dimension
    ///   - hiddenDim: Hidden layer dimension (typically 4x dim)
    public init(dim: Int, hiddenDim: Int) {
        self._fc1.wrappedValue = Linear(dim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, dim)
    }

    /// Forward pass
    /// - Parameter x: Input tensor [B, N, D]
    /// - Returns: Output tensor [B, N, D]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = gelu(x)
        x = fc2(x)
        return x
    }
}
