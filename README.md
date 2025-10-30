# Swift-AIM

Swift implementation of AIMv2 (Autoregressive Image Models v2) for Apple Silicon using MLX Swift.

## Overview

Swift-AIM provides a pure Swift implementation of Apple's AIMv2 Vision Transformer architecture for Apple Silicon devices. This library focuses on **inference only** and leverages [MLX Swift](https://github.com/ml-explore/mlx-swift) for optimal performance.

### Features

- ✅ Complete AIMv2 architecture implementation
- ✅ Support for multiple image resolutions (224×224, 336×336, 448×448)
- ✅ Absolute and sinusoidal positional embeddings
- ✅ PyTorch weight loading and sanitization
- ✅ Comprehensive test suite (48+ tests)
- ✅ Built with Swift 6.2 and strict concurrency
- ✅ Image preprocessing with ImageNet normalization
- ✅ Safetensors loader with inspection utilities
- ✅ High-level inference API
- ✅ HuggingFace Hub integration (basic structure)
- ✅ Configuration loading from JSON
- ✅ Batch processing support

## Requirements

- macOS 14.0+ / iOS 17.0+
- Swift 6.2+
- Xcode 16.0+ (for building)
- Apple Silicon (M1/M2/M3/M4)

## Installation

### Swift Package Manager

Add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/swift-aim", from: "1.0.0")
]
```

Or add it via Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select version/branch

## Quick Start

### Simple Inference

```swift
import SwiftAIM

// Load model from local directory containing config.json and model.safetensors
let aim = try AIMv2.load(fromPath: "/path/to/model")

// Print model information
print(aim.info)
// Output:
// AIMv2 Model Information
// Type: aimv2-large-patch14-224
// Image Size: 224×224
// ...

#if canImport(AppKit)
// Load and process image (macOS)
import AppKit
let image = NSImage(contentsOfFile: "photo.jpg")!

// Extract features
let features = aim.encode(image)
print("Features shape:", features?.shape)  // [1, 257, 768]

// Extract CLS token for classification
let clsFeature = aim.extractCLSFeature(image)
print("CLS feature:", clsFeature?.shape)  // [1, 768]
#endif
```

### Load from HuggingFace Cache

```swift
import SwiftAIM

// Try to load from cache
do {
    let aim = try HuggingFaceHub.load(modelID: "apple/aimv2-large-patch14-224")
    print("Model loaded successfully")
} catch HubError.modelNotCached(_, _, let instructions) {
    print("Model not cached. Download instructions:")
    print(instructions)
}
```

### Manual Setup with Preprocessing

```swift
import SwiftAIM

// 1. Load configuration from JSON
let config = try AIMv2Configuration.load(
    from: URL(fileURLWithPath: "/path/to/config.json")
)

// 2. Create model
let model = AIMv2Model(config: config)

// 3. Load and sanitize weights
let weights = try SafetensorsLoader.loadAndSanitize(
    fromPath: "/path/to/model.safetensors"
)

// 4. Apply weights
let unflattenedWeights = ModuleParameters.unflattened(weights)
try model.update(parameters: unflattenedWeights, verify: .none)
MLX.eval(model)

// 5. Create high-level API
let aim = AIMv2(config: config, model: model)

// 6. Use image preprocessor
let preprocessor = ImagePreprocessor(imageSize: 224)
#if canImport(CoreGraphics)
let cgImage: CGImage = loadImage()
let pixels = preprocessor.preprocess(cgImage)
let features = aim.encode(pixels: pixels)
#endif
```

## Supported Models

The architecture supports any AIMv2 model with the following configurations:

| Model | Parameters | Resolution | Patches | Architecture Status |
|-------|-----------|------------|---------|---------------------|
| aimv2-large-patch14-224 | 0.3B | 224×224 | 256 | ✅ Compatible |
| aimv2-large-patch14-336 | 0.3B | 336×336 | 576 | ✅ Compatible |
| aimv2-large-patch14-448 | 0.3B | 448×448 | 1024 | ✅ Compatible |
| aimv2-huge-patch14-224 | 0.7B | 224×224 | 256 | ✅ Compatible |
| aimv2-1B-patch14-224 | 1B | 224×224 | 256 | ✅ Compatible |
| aimv2-3B-patch14-224 | 3B | 224×224 | 256 | ✅ Compatible |

All models are available on [HuggingFace](https://huggingface.co/collections/apple/aimv2).

**Note**: Weight loading from HuggingFace Hub is not yet implemented. You need to manually download and load safetensors files.

## Architecture

Swift-AIM implements the AIMv2 Vision Transformer architecture:

```
Input Image [B, 3, H, W]
    ↓
Patch Embedding (14×14 patches)
    ↓
Add CLS Token
    ↓
Add Position Embedding
    ↓
Transformer Blocks (×12)
    ↓
Layer Normalization
    ↓
Output Features [B, N+1, D]
```

### Key Components

- **PatchEmbed**: Splits image into 14×14 patches using Conv2d
- **Attention**: Multi-head self-attention mechanism
- **MLP**: Feed-forward network with GELU activation
- **TransformerBlock**: Pre-norm architecture with residual connections

## Implementation Status

### ✅ Completed Features (v0.3.0)

- **Core Architecture**
  - AIMv2Model with full Vision Transformer implementation
  - PatchEmbed (14×14 patch embedding via Conv2d)
  - Multi-head self-attention
  - MLP with GELU activation
  - TransformerBlock with pre-norm architecture
  - LayerNorm with configurable epsilon

- **Position Embeddings**
  - Absolute learnable positional embeddings
  - Sinusoidal (sincos) positional embeddings with caching

- **Weight Management**
  - PyTorch → MLX weight conversion
  - Safetensors loader with MLX integration
  - Weight sanitization with validation
  - Conv2d weight transpose handling
  - Weight inspection utilities

- **Configuration**
  - Type-safe PositionEmbeddingType enum
  - Comprehensive configuration validation
  - Support for multiple image resolutions
  - Computed properties (numPatches, sequenceLength, headDim)
  - JSON loading from config.json files

- **Image Processing**
  - ImagePreprocessor with ImageNet normalization
  - Support for CGImage, NSImage (macOS), UIImage (iOS)
  - Resize and center crop
  - Batch preprocessing

- **High-Level API**
  - AIMv2 class for easy inference
  - Model loading from directory
  - CLS token and patch feature extraction
  - Batch processing support
  - Model information utilities

- **HuggingFace Hub**
  - Model cache management
  - Local cache inspection
  - Download instructions generation
  - Model repository handling

- **Testing**
  - 48+ comprehensive tests
  - Model architecture tests
  - Layer-level tests
  - Weight sanitization tests
  - Integration tests

### 🚧 Future Enhancements

- Automatic model downloading from HuggingFace Hub
- Performance benchmarking suite
- Metal Performance Shaders optimization
- Quantization support
- Example applications

## Code Examples

### Weight Sanitization

```swift
import SwiftAIM

// Load weights from safetensors (you need to implement this)
let rawWeights: [String: MLXArray] = loadSafetensors(from: "model.safetensors")

// Sanitize PyTorch weights to MLX format
let sanitizedWeights = AIMv2Model.sanitize(weights: rawWeights)

// The sanitization:
// 1. Removes "vision_model." and "encoder." prefixes
// 2. Transposes Conv2d weights: [out, in, h, w] → [out, h, w, in]
// 3. Validates weight dimensions and sizes
// 4. Skips invalid or malformed weights

// Apply to model
model.update(parameters: sanitizedWeights)
```

### Using Different Position Embeddings

```swift
// Absolute (learnable) position embeddings
let configAbsolute = AIMv2Configuration(
    modelType: "aimv2-test",
    hiddenSize: 768,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    intermediateSize: 3072,
    imageSize: 224,
    positionEmbeddingType: .absolute
)

// Sinusoidal (fixed) position embeddings
let configSincos = AIMv2Configuration(
    modelType: "aimv2-test",
    hiddenSize: 768,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    intermediateSize: 3072,
    imageSize: 224,
    positionEmbeddingType: .sincos  // More efficient for inference
)
```

### Batch Processing

```swift
// Create batch of images
let batchSize = 4
let pixels = MLXRandom.normal([batchSize, 3, 224, 224])

// Forward pass
let features = model(pixels)  // [4, 257, 768]

// Extract features for each image in batch
for i in 0..<batchSize {
    let imageFeature = features[i, 0, 0...]  // CLS feature for image i
    print("Image \(i) feature shape:", imageFeature.shape)
}
```

## Documentation

- [USAGE.md](USAGE.md) - Comprehensive usage guide with examples
- [CLAUDE.md](CLAUDE.md) - Development guide and architecture overview
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Code review and improvement plan
- [CHANGELOG.md](CHANGELOG.md) - Detailed change history

## Development

### Building

```bash
# Build the package (command line)
swift build

# Open in Xcode (recommended for Metal shader support)
open Package.swift
# or
xcodebuild -scheme swift-aim
```

**Note**: Metal shaders cannot be built by SwiftPM from command line. For final builds and testing, use Xcode.

### Testing

Swift-AIM uses **Swift Testing** framework (not XCTest):

```bash
# Run all tests (from Xcode recommended)
swift test

# Run specific test suite
swift test --filter "AIMv2Model Tests"

# Run specific test
swift test --filter "testModelInitialization"
```

#### Test Suites

- **AIMv2ModelTests** (13 tests)
  - Model initialization and configuration
  - Forward pass shape validation
  - CLS and patch feature extraction
  - Position embedding modes
  - Multiple image resolutions
  - Input validation

- **LayerTests** (20+ tests)
  - PatchEmbed output shapes and calculations
  - Attention mechanism (scale, heads, sequences)
  - MLP dimension preservation
  - TransformerBlock residual connections
  - Layer integration tests

- **WeightSanitizationTests** (15 tests)
  - Prefix removal (vision_model, encoder)
  - Conv2d weight transpose validation
  - Weight validation and error handling
  - Complete model weight sanitization

Total: **48+ comprehensive logic-based tests**

## Project Status

**Current Version**: v0.3.0 (Full-Featured Release)

### Recent Updates (2025-10-30)

**Phase 1: Core Implementation (v0.2.0)**
- ✅ Complete AIMv2 architecture implementation
- ✅ Comprehensive test suite (48+ tests)
- ✅ Code quality improvements (Phase 1 & 2)
- ✅ All documentation translated to English
- ✅ Weight sanitization with validation
- ✅ Type-safe configuration

**Phase 2: Full Features (v0.3.0)**
- ✅ ImagePreprocessor with ImageNet normalization
- ✅ Safetensors loader with MLX integration
- ✅ High-level AIMv2 inference API
- ✅ Configuration loading from JSON
- ✅ HuggingFace Hub integration (basic structure)
- ✅ Comprehensive usage documentation (USAGE.md)
- ✅ Batch processing support

### Next Steps

1. Automatic model downloading from HuggingFace Hub
2. Performance benchmarking suite
3. Example applications (image similarity, classification)
4. Metal Performance Shaders optimization
5. Quantization support

## Quality Metrics

- **Test Coverage**: 48+ comprehensive tests
- **Code Quality**: 10+ precondition validations
- **Type Safety**: Enum-based configuration
- **Performance**: Sincos position embedding caching
- **Security**: Weight sanitization with size validation

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass (`swift test`)
2. Code follows Swift 6.2 conventions
3. Documentation is updated
4. Strict concurrency compliance

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Apple's [ml-aim](https://github.com/apple/ml-aim) for the original AIMv2 models and research
- [MLX Swift](https://github.com/ml-explore/mlx-swift) for the ML framework on Apple Silicon
- [HuggingFace](https://huggingface.co) for model hosting

## Citation

For the original AIMv2 work:

```bibtex
@article{aimv2,
  title={Autoregressive Image Models},
  author={Apple ML Research},
  journal={arXiv},
  year={2024}
}
```

## Related Projects

- [ml-aim](https://github.com/apple/ml-aim) - Original PyTorch implementation
- [mlx-swift](https://github.com/ml-explore/mlx-swift) - MLX Swift framework
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - MLX Swift examples
- [DeepWiki: ml-aim](https://deepwiki.com/apple/ml-aim) - Documentation and research

---

**Development Status**: Core implementation complete, additional features in progress.
