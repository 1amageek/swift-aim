import MLX
import Foundation

/// Safetensors weight file loader
///
/// Provides utilities for loading model weights from safetensors format files.
public struct SafetensorsLoader {

    /// Load weights from a safetensors file
    /// - Parameter url: URL to the safetensors file
    /// - Returns: Dictionary of weight name to MLXArray
    /// - Throws: LoadingError if file cannot be loaded
    public static func load(from url: URL) throws -> [String: MLXArray] {
        // Check file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadingError.fileNotFound(url.path)
        }

        // MLX provides direct safetensors loading
        do {
            let weights = try MLX.loadArrays(url: url)
            return weights
        } catch {
            throw LoadingError.invalidFormat("Failed to load safetensors: \(error.localizedDescription)")
        }
    }

    /// Load weights from a safetensors file at path
    /// - Parameter path: File path to the safetensors file
    /// - Returns: Dictionary of weight name to MLXArray
    /// - Throws: LoadingError if file cannot be loaded
    public static func load(fromPath path: String) throws -> [String: MLXArray] {
        let url = URL(fileURLWithPath: path)
        return try load(from: url)
    }

    /// Load and sanitize weights for AIMv2 model
    /// - Parameter url: URL to the safetensors file
    /// - Returns: Sanitized weights ready for model.update()
    /// - Throws: LoadingError if file cannot be loaded
    public static func loadAndSanitize(from url: URL) throws -> [String: MLXArray] {
        let rawWeights = try load(from: url)
        return AIMv2Model.sanitize(weights: rawWeights)
    }

    /// Load and sanitize weights for AIMv2 model from path
    /// - Parameter path: File path to the safetensors file
    /// - Returns: Sanitized weights ready for model.update()
    /// - Throws: LoadingError if file cannot be loaded
    public static func loadAndSanitize(fromPath path: String) throws -> [String: MLXArray] {
        let url = URL(fileURLWithPath: path)
        return try loadAndSanitize(from: url)
    }

    /// List all tensor names in a safetensors file without loading data
    /// - Parameter url: URL to the safetensors file
    /// - Returns: Array of tensor names
    /// - Throws: LoadingError if file cannot be read
    public static func listTensors(in url: URL) throws -> [String] {
        // Load the weights (MLX is efficient with memory mapping)
        let weights = try load(from: url)
        return Array(weights.keys).sorted()
    }

    /// Get information about tensors in a safetensors file
    /// - Parameter url: URL to the safetensors file
    /// - Returns: Dictionary of tensor name to shape and dtype info
    /// - Throws: LoadingError if file cannot be read
    public static func inspectTensors(in url: URL) throws -> [String: TensorInfo] {
        let weights = try load(from: url)
        var info: [String: TensorInfo] = [:]

        for (name, array) in weights {
            info[name] = TensorInfo(
                name: name,
                shape: array.shape,
                dtype: array.dtype,
                size: array.size
            )
        }

        return info
    }
}

/// Tensor information
public struct TensorInfo: CustomStringConvertible {
    public let name: String
    public let shape: [Int]
    public let dtype: DType
    public let size: Int

    public var description: String {
        let shapeStr = shape.map(String.init).joined(separator: " × ")
        return "\(name): [\(shapeStr)] \(dtype) (\(size) elements)"
    }
}

/// Weight loading errors
public enum LoadingError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidFormat(String)
    case unsupportedVersion(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid format: \(message)"
        case .unsupportedVersion(let version):
            return "Unsupported version: \(version)"
        }
    }
}
