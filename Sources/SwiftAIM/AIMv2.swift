import MLX
import MLXNN
import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// High-level AIMv2 inference API
///
/// Provides a simple interface for loading models and running inference.
///
/// Example:
/// ```swift
/// // Load model from directory
/// let aim = try AIMv2.load(from: "/path/to/model")
///
/// // Run inference
/// let image = loadImage()
/// let features = try aim.encode(image)
/// ```
public class AIMv2 {

    /// Model configuration
    public let config: AIMv2Configuration

    /// The underlying model
    public let model: AIMv2Model

    /// Image preprocessor
    public let preprocessor: ImagePreprocessor

    /// Initialize with configuration and model
    /// - Parameters:
    ///   - config: Model configuration
    ///   - model: Pre-initialized model
    public init(config: AIMv2Configuration, model: AIMv2Model) {
        self.config = config
        self.model = model
        self.preprocessor = ImagePreprocessor(imageSize: config.imageSize)
    }

    /// Load model from directory containing config.json and weights
    /// - Parameters:
    ///   - directory: Directory containing model files
    ///   - weightsFile: Name of weights file (default: "model.safetensors")
    /// - Returns: Loaded AIMv2 instance
    /// - Throws: LoadingError if files cannot be loaded
    public static func load(
        from directory: URL,
        weightsFile: String = "model.safetensors"
    ) throws -> AIMv2 {
        // Load configuration
        let configURL = directory.appendingPathComponent("config.json")
        let config = try AIMv2Configuration.load(from: configURL)

        // Initialize model
        let model = AIMv2Model(config: config)

        // Load weights
        let weightsURL = directory.appendingPathComponent(weightsFile)
        let weights = try SafetensorsLoader.loadAndSanitize(from: weightsURL)

        // Update model with weights
        let unflattenedWeights = ModuleParameters.unflattened(weights)
        try model.update(parameters: unflattenedWeights, verify: .all)

        // Set to eval mode
        MLX.eval(model)

        return AIMv2(config: config, model: model)
    }

    /// Load model from directory path
    /// - Parameters:
    ///   - path: Directory path containing model files
    ///   - weightsFile: Name of weights file (default: "model.safetensors")
    /// - Returns: Loaded AIMv2 instance
    /// - Throws: LoadingError if files cannot be loaded
    public static func load(
        fromPath path: String,
        weightsFile: String = "model.safetensors"
    ) throws -> AIMv2 {
        let url = URL(fileURLWithPath: path)
        return try load(from: url, weightsFile: weightsFile)
    }

    // MARK: - Inference Methods

    #if canImport(CoreGraphics)

    /// Encode image to features
    /// - Parameter image: Input CGImage
    /// - Returns: Feature tensor [1, N+1, D], or nil if preprocessing fails
    public func encode(_ image: CGImage) -> MLXArray? {
        guard let pixels = preprocessor.preprocess(image) else {
            return nil
        }
        return model(pixels)
    }

    /// Extract CLS token feature from image
    /// - Parameter image: Input CGImage
    /// - Returns: CLS feature [1, D], or nil if preprocessing fails
    public func extractCLSFeature(_ image: CGImage) -> MLXArray? {
        guard let pixels = preprocessor.preprocess(image) else {
            return nil
        }
        return model.extractCLSFeature(pixels)
    }

    /// Extract patch features from image
    /// - Parameter image: Input CGImage
    /// - Returns: Patch features [1, N, D], or nil if preprocessing fails
    public func extractPatchFeatures(_ image: CGImage) -> MLXArray? {
        guard let pixels = preprocessor.preprocess(image) else {
            return nil
        }
        return model.extractPatchFeatures(pixels)
    }

    #endif

    /// Encode preprocessed pixels to features
    /// - Parameter pixels: Preprocessed pixel tensor [B, H, W, C] (channels-last, MLX format)
    /// - Returns: Feature tensor [B, N+1, D]
    public func encode(pixels: MLXArray) -> MLXArray {
        return model(pixels)
    }

    /// Extract CLS token feature from preprocessed pixels
    /// - Parameter pixels: Preprocessed pixel tensor [B, H, W, C] (channels-last, MLX format)
    /// - Returns: CLS feature [B, D]
    public func extractCLSFeature(pixels: MLXArray) -> MLXArray {
        return model.extractCLSFeature(pixels)
    }

    /// Extract patch features from preprocessed pixels
    /// - Parameter pixels: Preprocessed pixel tensor [B, H, W, C] (channels-last, MLX format)
    /// - Returns: Patch features [B, N, D]
    public func extractPatchFeatures(pixels: MLXArray) -> MLXArray {
        return model.extractPatchFeatures(pixels)
    }

    // MARK: - Batch Processing

    /// Encode multiple images in batch
    /// - Parameter images: Array of CGImages
    /// - Returns: Feature tensor [B, N+1, D], or nil if any preprocessing fails
    #if canImport(CoreGraphics)
    public func encodeBatch(_ images: [CGImage]) -> MLXArray? {
        guard let batch = preprocessor.batchPreprocess(images) else {
            return nil
        }
        return model(batch)
    }
    #endif

    /// Extract CLS features from multiple images
    /// - Parameter images: Array of CGImages
    /// - Returns: CLS features [B, D], or nil if any preprocessing fails
    #if canImport(CoreGraphics)
    public func extractCLSFeaturesBatch(_ images: [CGImage]) -> MLXArray? {
        guard let batch = preprocessor.batchPreprocess(images) else {
            return nil
        }
        return model.extractCLSFeature(batch)
    }
    #endif

    // MARK: - Model Information

    /// Get model information summary
    public var info: ModelInfo {
        ModelInfo(
            modelType: config.modelType,
            imageSize: config.imageSize,
            patchSize: config.patchSize,
            hiddenSize: config.hiddenSize,
            numLayers: config.numHiddenLayers,
            numHeads: config.numAttentionHeads,
            numPatches: config.numPatches,
            sequenceLength: config.sequenceLength,
            positionEmbeddingType: config.positionEmbeddingType
        )
    }
}

/// Model information
public struct ModelInfo: CustomStringConvertible {
    public let modelType: String
    public let imageSize: Int
    public let patchSize: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numPatches: Int
    public let sequenceLength: Int
    public let positionEmbeddingType: PositionEmbeddingType

    public var description: String {
        """
        AIMv2 Model Information
        ----------------------
        Type: \(modelType)
        Image Size: \(imageSize)×\(imageSize)
        Patch Size: \(patchSize)×\(patchSize)
        Patches: \(numPatches)
        Sequence Length: \(sequenceLength) (patches + CLS)
        Hidden Size: \(hiddenSize)
        Layers: \(numLayers)
        Attention Heads: \(numHeads)
        Position Embedding: \(positionEmbeddingType)
        """
    }
}

#if canImport(AppKit)
import AppKit

extension AIMv2 {
    /// Encode NSImage (macOS)
    public func encode(_ image: NSImage) -> MLXArray? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        return encode(cgImage)
    }

    /// Extract CLS feature from NSImage (macOS)
    public func extractCLSFeature(_ image: NSImage) -> MLXArray? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        return extractCLSFeature(cgImage)
    }
}
#endif

#if canImport(UIKit)
import UIKit

extension AIMv2 {
    /// Encode UIImage (iOS)
    public func encode(_ image: UIImage) -> MLXArray? {
        guard let cgImage = image.cgImage else {
            return nil
        }
        return encode(cgImage)
    }

    /// Extract CLS feature from UIImage (iOS)
    public func extractCLSFeature(_ image: UIImage) -> MLXArray? {
        guard let cgImage = image.cgImage else {
            return nil
        }
        return extractCLSFeature(cgImage)
    }
}
#endif
