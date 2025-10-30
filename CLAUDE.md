# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**swift-aim** is a Swift library for running AIMv2 (Autoregressive Image Models v2) inference on Apple Silicon using MLX Swift. The goal is to load pre-trained AIMv2 models from HuggingFace and perform image feature extraction.

- **Inference only** - No training capabilities required
- **Swift 6.2** with strict concurrency enabled
- **MLX Swift 0.10.0+** for Apple Silicon optimization
- **HuggingFace integration** for model loading

Target models: `apple/aimv2-large-patch14-224`, `apple/aimv2-large-patch14-336`, and other AIMv2 variants.

## Development Commands

### Building
```bash
swift build
```

Note: Metal shaders cannot be built by SwiftPM from command line. For final builds, use Xcode or xcodebuild:
```bash
xcodebuild -scheme swift-aim
```

### Testing

This project uses **Swift Testing** framework (Swift 6.0+), not XCTest.

```bash
# Run all tests
swift test

# Run specific test by name
swift test --filter SwiftAIMTests.testModelLoading

# Run tests in a specific suite
swift test --filter SwiftAIMTests
```

**Test Structure**:
- Use `@Test` macro instead of XCTest's `func test...`
- Use `#expect(...)` for assertions instead of `XCTAssert...`
- Async tests are supported with `async throws`

### Xcode
```bash
open Package.swift
```

## Architecture

### Core Design Principles

1. **Module-based architecture**: All neural network components inherit from `Module` (MLX Swift)
2. **Weight reuse**: Model structure is defined in Swift, but pre-trained weights are loaded from HuggingFace safetensors files
3. **No training**: Models are inference-only with eval mode fixed
4. **Weight sanitization**: PyTorch tensors need conversion to MLX format (e.g., Conv2d weight transposition)

### Planned Module Structure

```
SwiftAIM/
├── Core/
│   ├── AIMv2Model.swift          # Main vision encoder
│   └── AIMv2Configuration.swift  # Model config from HF
├── Layers/
│   ├── PatchEmbed.swift          # 14x14 patch embedding via Conv2d
│   ├── Attention.swift           # Multi-head self-attention
│   ├── MLP.swift                 # Feed-forward network
│   └── TransformerBlock.swift    # Pre-norm transformer block
├── Loading/
│   ├── ModelLoader.swift         # HuggingFace Hub integration
│   ├── WeightLoader.swift        # Safetensors parsing
│   └── WeightConverter.swift     # PyTorch → MLX conversion
└── Processing/
    └── ImagePreprocessor.swift   # ImageNet normalization
```

### Key Implementation Details

**AIMv2 Architecture**:
- Vision Transformer with 14x14 patch size (fixed)
- Pre-norm architecture: `x = x + Attention(LayerNorm(x))`
- CLS token prepended to patch sequence
- Absolute or sincos positional embeddings
- Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**MLX Swift Patterns**:

### Module System
All neural network layers must subclass `Module` from MLXNN. The `Module` class provides automatic parameter discovery, training state management, and hierarchical composition via Swift's reflection system.

### Property Wrappers
**CRITICAL**: Property wrapper initialization must use `self._propertyName.wrappedValue = ...` syntax:

```swift
// ✅ CORRECT: Use wrappedValue for initialization
public class MyLayer: Module {
    @ModuleInfo public var linear: Linear
    @ParameterInfo public var bias: MLXArray

    public init(dim: Int) {
        self._linear.wrappedValue = Linear(dim, dim)
        self._bias.wrappedValue = MLXArray.zeros([dim])
    }
}

// ❌ WRONG: Direct assignment fails or causes runtime errors
public init(dim: Int) {
    self.linear = Linear(dim, dim)  // This is incorrect!
    self.bias = MLXArray.zeros([dim])  // This is incorrect!
}
```

**Why**: Direct assignment to wrapped properties after initialization triggers `fatalError`. The `wrappedValue` syntax ensures proper cache updates in the `Module` system. After initialization, use `Module.update(parameters:)` or `Module.update(modules:)` for updates.

### @ModuleInfo Usage
- Used for child `Module` instances (e.g., `Linear`, `Conv2d`, `LayerNorm`)
- Enables quantization and dynamic module replacement
- Required for `Module.update(modules:)` to work
- Can specify custom keys: `@ModuleInfo(key: "custom_key")`

### @ParameterInfo Usage
- Used for `MLXArray` parameters (weights, biases, embeddings)
- Enables parameter discovery via `Module.parameters()`
- Required for `Module.update(parameters:)` to work
- Can specify custom keys: `@ParameterInfo(key: "custom_key")`

### Conv2d and IntOrPair
Conv2d expects input in `NHWC` format (batch, height, width, channels):

```swift
// Initialize Conv2d with IntOrPair
self._projection.wrappedValue = Conv2d(
    inputChannels: 3,
    outputChannels: 768,
    kernelSize: IntOrPair(14),      // Same for both dimensions
    stride: IntOrPair(14),           // Same for both dimensions
    padding: IntOrPair(0)            // No padding
)

// Or with different values per dimension
kernelSize: IntOrPair([14, 14])    // [height, width]
```

`IntOrPair` supports:
- `IntOrPair(5)` - Same value for both dimensions
- `IntOrPair([3, 5])` - Different values [height, width]
- `IntOrPair((3, 5))` - Tuple initialization

### Forward Pass
Implement `callAsFunction(_ x: MLXArray) -> MLXArray` for forward computation:

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = layer1(x)
    x = activation(x)
    x = layer2(x)
    return x
}
```

### Inference Mode
Call `eval(model)` after loading weights to set all layers to inference mode (disables dropout, etc.):

```swift
let model = AIMv2Model(config: config)
model.update(parameters: loadedWeights)
eval(model)  // Set to inference mode
```

**Weight Loading Flow**:
1. Download model files from HuggingFace (config.json, *.safetensors)
2. Parse config.json into `AIMv2Configuration`
3. Instantiate model with random weights
4. Load safetensors into `[String: MLXArray]` dictionary
5. Sanitize weights (transpose Conv2d: `[out, in, h, w]` → `[out, h, w, in]`)
6. Apply weights via `model.update(parameters:)`
7. Call `eval(model)` for inference mode

**Swift 6.2 Concurrency**:
- Mark configurations and data types as `Sendable`
- Use `actor` for ModelLoader to ensure thread-safety
- Enable strict concurrency checking in Package.swift

## Dependencies

Currently no dependencies - MLX Swift will be added:

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
]
```

Required products: `MLX`, `MLXNN`, `MLXFast`

## Reference Resources

- AIMv2 paper/repo: https://github.com/apple/ml-aim
- MLX Swift: https://github.com/ml-explore/mlx-swift
- MLX Swift Examples: https://github.com/ml-explore/mlx-swift-examples
- HuggingFace models: https://huggingface.co/collections/apple/aimv2
- DeepWiki (for research):
  - AIMv2: https://deepwiki.com/apple/ml-aim
  - MLX Swift: https://deepwiki.com/ml-explore/mlx-swift
