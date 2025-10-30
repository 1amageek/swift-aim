import Testing
import Foundation
@testable import SwiftAIM

@Test func testConfigurationDecoding() async throws {
    let json = """
    {
        "model_type": "aimv2-large-patch14-224",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "image_size": 224,
        "patch_size": 14,
        "num_channels": 3,
        "layer_norm_eps": 1e-6,
        "position_embedding_type": "absolute",
        "qkv_bias": true,
        "mlp_ratio": 4.0
    }
    """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(AIMv2Configuration.self, from: data)

    #expect(config.hiddenSize == 768)
    #expect(config.numHiddenLayers == 12)
    #expect(config.numAttentionHeads == 12)
    #expect(config.imageSize == 224)
    #expect(config.patchSize == 14)
    #expect(config.numPatches == 256)
    #expect(config.sequenceLength == 257)
}

@Test func testConfigurationInit() {
    let config = AIMv2Configuration(
        modelType: "test-model",
        hiddenSize: 768,
        numHiddenLayers: 12,
        numAttentionHeads: 12,
        intermediateSize: 3072,
        imageSize: 224
    )

    #expect(config.modelType == "test-model")
    #expect(config.hiddenSize == 768)
    #expect(config.headDim == 64)  // 768 / 12
}
