import Foundation

/// Position embedding type for the model
public enum PositionEmbeddingType: String, Codable, Sendable {
    /// Learnable absolute positional embeddings
    case absolute
    /// Fixed sinusoidal positional embeddings
    case sincos
}

/// AIMv2 model configuration
///
/// Holds model configuration loaded from HuggingFace `config.json`.
public struct AIMv2Configuration: Codable, Sendable {
    /// Model type (e.g., "aimv2-large-patch14-224")
    public let modelType: String

    /// Embedding dimension (e.g., 768)
    public let hiddenSize: Int

    /// Number of transformer layers (e.g., 12)
    public let numHiddenLayers: Int

    /// Number of attention heads (e.g., 12)
    public let numAttentionHeads: Int

    /// MLP intermediate dimension (e.g., 3072)
    public let intermediateSize: Int

    /// Input image size in pixels (e.g., 224, 336, 448)
    public let imageSize: Int

    /// Patch size (fixed at 14)
    public let patchSize: Int

    /// Number of input channels (RGB=3)
    public let numChannels: Int

    /// LayerNorm epsilon value
    public let layerNormEps: Float

    /// Type of positional embedding
    public let positionEmbeddingType: PositionEmbeddingType

    /// Whether to use bias in QKV projection
    public let qkvBias: Bool

    /// MLP ratio (hiddenSize * mlpRatio = intermediateSize)
    public let mlpRatio: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case layerNormEps = "layer_norm_eps"
        case positionEmbeddingType = "position_embedding_type"
        case qkvBias = "qkv_bias"
        case mlpRatio = "mlp_ratio"
    }

    /// Initializer
    public init(
        modelType: String,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        intermediateSize: Int,
        imageSize: Int,
        patchSize: Int = 14,
        numChannels: Int = 3,
        layerNormEps: Float = 1e-6,
        positionEmbeddingType: PositionEmbeddingType = .absolute,
        qkvBias: Bool = true,
        mlpRatio: Float = 4.0
    ) {
        // Validation: Positive value checks
        precondition(hiddenSize > 0, "hiddenSize must be positive, got \(hiddenSize)")
        precondition(numHiddenLayers > 0, "numHiddenLayers must be positive, got \(numHiddenLayers)")
        precondition(numAttentionHeads > 0, "numAttentionHeads must be positive, got \(numAttentionHeads)")
        precondition(intermediateSize > 0, "intermediateSize must be positive, got \(intermediateSize)")
        precondition(imageSize > 0, "imageSize must be positive, got \(imageSize)")
        precondition(patchSize > 0, "patchSize must be positive, got \(patchSize)")
        precondition(numChannels > 0, "numChannels must be positive, got \(numChannels)")

        // Validation: Dimension consistency checks
        precondition(hiddenSize % numAttentionHeads == 0,
                     "hiddenSize (\(hiddenSize)) must be divisible by numAttentionHeads (\(numAttentionHeads))")
        precondition(imageSize % patchSize == 0,
                     "imageSize (\(imageSize)) must be divisible by patchSize (\(patchSize))")

        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.layerNormEps = layerNormEps
        self.positionEmbeddingType = positionEmbeddingType
        self.qkvBias = qkvBias
        self.mlpRatio = mlpRatio
    }

    /// Calculate the number of patches
    public var numPatches: Int {
        (imageSize / patchSize) * (imageSize / patchSize)
    }

    /// Calculate the sequence length (number of patches + CLS token)
    public var sequenceLength: Int {
        numPatches + 1
    }

    /// Dimension per attention head
    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }

    /// Load configuration from JSON file
    /// - Parameter url: URL to config.json file
    /// - Returns: Decoded AIMv2Configuration
    /// - Throws: DecodingError if JSON is invalid
    public static func load(from url: URL) throws -> AIMv2Configuration {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(AIMv2Configuration.self, from: data)
    }

    /// Load configuration from JSON data
    /// - Parameter data: JSON data
    /// - Returns: Decoded AIMv2Configuration
    /// - Throws: DecodingError if JSON is invalid
    public static func load(from data: Data) throws -> AIMv2Configuration {
        let decoder = JSONDecoder()
        return try decoder.decode(AIMv2Configuration.self, from: data)
    }

    /// Load configuration from JSON string
    /// - Parameter json: JSON string
    /// - Returns: Decoded AIMv2Configuration
    /// - Throws: DecodingError if JSON is invalid
    public static func load(fromJSON json: String) throws -> AIMv2Configuration {
        guard let data = json.data(using: .utf8) else {
            throw ConfigurationError.invalidJSON("Failed to encode JSON string to UTF-8")
        }
        return try load(from: data)
    }
}

/// Configuration loading errors
public enum ConfigurationError: Error {
    case invalidJSON(String)
    case fileNotFound(String)
}
