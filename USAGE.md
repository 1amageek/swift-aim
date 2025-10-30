# Usage Guide

Comprehensive examples for using Swift-AIM.

## Table of Contents

- [Quick Start](#quick-start)
- [Loading Models](#loading-models)
- [Image Preprocessing](#image-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Batch Processing](#batch-processing)
- [Advanced Usage](#advanced-usage)

## Quick Start

### Simple Inference

```swift
import SwiftAIM

// Load model from local directory
let aim = try AIMv2.load(fromPath: "/path/to/model")

// Print model information
print(aim.info)

// Load and preprocess image (macOS)
#if canImport(AppKit)
import AppKit
let image = NSImage(contentsOfFile: "photo.jpg")!
let features = aim.encode(image)
print("Features shape:", features.shape)  // [1, 257, 768]
#endif
```

## Loading Models

### From Local Directory

```swift
// Directory should contain:
// - config.json
// - model.safetensors

let aim = try AIMv2.load(fromPath: "/Users/me/models/aimv2-large-patch14-224")
```

### From HuggingFace Cache

```swift
import SwiftAIM

// Try to load from cache
do {
    let aim = try HuggingFaceHub.load(modelID: "apple/aimv2-large-patch14-224")
    print("Model loaded from cache")
} catch HubError.modelNotCached(let modelID, _, let instructions) {
    print("Model not cached. Download instructions:")
    print(instructions)
}
```

### Manual Model Setup

```swift
import SwiftAIM
import MLX

// 1. Load configuration
let configURL = URL(fileURLWithPath: "/path/to/config.json")
let config = try AIMv2Configuration.load(from: configURL)

// 2. Create model
let model = AIMv2Model(config: config)

// 3. Load weights
let weights = try SafetensorsLoader.loadAndSanitize(
    fromPath: "/path/to/model.safetensors"
)

// 4. Apply weights
model.update(parameters: ModuleParameters.unflattened(weights))
MLX.eval(model)

// 5. Create high-level API
let aim = AIMv2(config: config, model: model)
```

## Image Preprocessing

### Basic Preprocessing

```swift
let preprocessor = ImagePreprocessor(imageSize: 224)

// From CGImage
#if canImport(CoreGraphics)
let cgImage: CGImage = ...
let pixels = preprocessor.preprocess(cgImage)
print("Preprocessed shape:", pixels.shape)  // [1, 3, 224, 224]
#endif
```

### Custom Normalization

```swift
// Use custom mean and std values
let preprocessor = ImagePreprocessor(
    imageSize: 224,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5]
)
```

### From Raw Pixel Data

```swift
import MLXRandom

// Create dummy image [H, W, C] in range [0, 1]
let rawPixels = MLXRandom.uniform([224, 224, 3])

let preprocessor = ImagePreprocessor(imageSize: 224)
let normalized = preprocessor.preprocess(pixels: rawPixels)
print("Output shape:", normalized.shape)  // [1, 3, 224, 224]
```

## Feature Extraction

### CLS Token Feature (for Classification)

```swift
import SwiftAIM

let aim = try AIMv2.load(fromPath: "/path/to/model")

#if canImport(AppKit)
let image = NSImage(contentsOfFile: "cat.jpg")!

// Extract CLS token feature
let clsFeature = aim.extractCLSFeature(image)
print("CLS feature shape:", clsFeature?.shape)  // [1, 768]

// Use for classification (requires separate classifier head)
// let logits = classifier(clsFeature)
// let predictedClass = logits.argmax()
#endif
```

### Patch Features (for Dense Prediction)

```swift
// Extract spatial features
let patchFeatures = aim.extractPatchFeatures(image)
print("Patch features shape:", patchFeatures?.shape)  // [1, 256, 768]

// Can be used for:
// - Object detection
// - Semantic segmentation
// - Dense captioning
```

### Full Feature Tensor

```swift
// Get all features (CLS + patches)
let allFeatures = aim.encode(image)
print("All features shape:", allFeatures?.shape)  // [1, 257, 768]

// Split manually
let cls = allFeatures?[0, 0, 0...]      // CLS token
let patches = allFeatures?[0, 1..., 0...]  // Patch tokens
```

## Batch Processing

### Process Multiple Images

```swift
import SwiftAIM

let aim = try AIMv2.load(fromPath: "/path/to/model")

#if canImport(AppKit)
let images = [
    NSImage(contentsOfFile: "img1.jpg")!,
    NSImage(contentsOfFile: "img2.jpg")!,
    NSImage(contentsOfFile: "img3.jpg")!
]

// Extract CLS features for all images
let features = aim.extractCLSFeaturesBatch(
    images.compactMap { $0.cgImage(forProposedRect: nil, context: nil, hints: nil) }
)
print("Batch features shape:", features.shape)  // [3, 768]

// Calculate similarity between images
let feat1 = features[0]
let feat2 = features[1]

let similarity = cosineSimilarity(feat1, feat2)
print("Similarity:", similarity)
#endif
```

### Manual Batch Processing

```swift
let preprocessor = ImagePreprocessor(imageSize: 224)

// Load images
let cgImages: [CGImage] = loadImages()

// Preprocess each
let preprocessed = cgImages.map { preprocessor.preprocess($0) }

// Concatenate into batch
let batch = MLX.concatenated(preprocessed, axis: 0)
print("Batch shape:", batch.shape)  // [N, 3, 224, 224]

// Run inference
let features = aim.encode(pixels: batch)
print("Features shape:", features.shape)  // [N, 257, 768]
```

## Advanced Usage

### Different Image Sizes

```swift
// 224×224 (fastest)
let aim224 = try AIMv2.load(fromPath: "/path/to/aimv2-large-patch14-224")

// 336×336 (better accuracy)
let aim336 = try AIMv2.load(fromPath: "/path/to/aimv2-large-patch14-336")

// 448×448 (best accuracy)
let aim448 = try AIMv2.load(fromPath: "/path/to/aimv2-large-patch14-448")
```

### Position Embedding Types

```swift
// Absolute (learnable) - default
let configAbs = AIMv2Configuration(
    modelType: "aimv2-test",
    hiddenSize: 768,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    intermediateSize: 3072,
    imageSize: 224,
    positionEmbeddingType: .absolute
)

// Sinusoidal (fixed) - more efficient
let configSincos = AIMv2Configuration(
    modelType: "aimv2-test",
    hiddenSize: 768,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    intermediateSize: 3072,
    imageSize: 224,
    positionEmbeddingType: .sincos
)
```

### Inspect Model Weights

```swift
// List all tensors in safetensors file
let tensorNames = try SafetensorsLoader.listTensors(
    in: URL(fileURLWithPath: "/path/to/model.safetensors")
)
print("Tensors:", tensorNames)

// Get detailed information
let info = try SafetensorsLoader.inspectTensors(
    in: URL(fileURLWithPath: "/path/to/model.safetensors")
)

for (name, tensorInfo) in info {
    print(tensorInfo)
}
```

### Weight Sanitization

```swift
// Load raw PyTorch weights
let rawWeights = try SafetensorsLoader.load(
    fromPath: "/path/to/pytorch_model.safetensors"
)

// Sanitize for MLX
let sanitized = AIMv2Model.sanitize(weights: rawWeights)

// Sanitization performs:
// 1. Remove "vision_model." and "encoder." prefixes
// 2. Transpose Conv2d weights: [out, in, h, w] → [out, h, w, in]
// 3. Validate weight dimensions
// 4. Skip invalid weights

print("Original keys:", rawWeights.count)
print("Sanitized keys:", sanitized.count)
```

### Custom Model Configuration

```swift
// Create custom configuration
let customConfig = AIMv2Configuration(
    modelType: "custom-aimv2",
    hiddenSize: 512,          // Smaller model
    numHiddenLayers: 6,       // Fewer layers
    numAttentionHeads: 8,
    intermediateSize: 2048,
    imageSize: 224,
    patchSize: 14,
    qkvBias: true
)

let customModel = AIMv2Model(config: customConfig)
```

### Image Similarity Search

```swift
import SwiftAIM

let aim = try AIMv2.load(fromPath: "/path/to/model")

// Extract features from image database
var database: [(filename: String, feature: MLXArray)] = []

for filename in imageFilenames {
    let image = loadImage(filename)
    if let feature = aim.extractCLSFeature(image) {
        database.append((filename, feature))
    }
}

// Query with new image
let queryImage = loadImage("query.jpg")
let queryFeature = aim.extractCLSFeature(queryImage)!

// Find most similar
var similarities: [(filename: String, score: Float)] = []

for (filename, dbFeature) in database {
    let similarity = cosineSimilarity(queryFeature, dbFeature)
    similarities.append((filename, similarity))
}

// Sort by similarity
similarities.sort { $0.score > $1.score }

print("Most similar images:")
for (filename, score) in similarities.prefix(5) {
    print("\(filename): \(score)")
}
```

### Helper Functions

```swift
// Cosine similarity
func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
    let dotProduct = (a * b).sum()
    let normA = sqrt((a * a).sum())
    let normB = sqrt((b * b).sum())
    return (dotProduct / (normA * normB)).item(Float.self)
}

// L2 normalization
func l2Normalize(_ x: MLXArray) -> MLXArray {
    let norm = sqrt((x * x).sum())
    return x / norm
}
```

## Troubleshooting

### Metal Library Error

If you see "Failed to load the default metallib" when running tests from command line:

```bash
# Use Xcode instead
open Package.swift
# Then run tests from Xcode (⌘U)
```

### Model Not Found

```swift
// Check if model is cached
let repo = HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large-patch14-224")
let cached = HuggingFaceHub.isCached(repo)
print("Is cached:", cached)

// Get cache path
let cachePath = HuggingFaceHub.localCachePath(for: repo)
print("Cache path:", cachePath.path)

// List cached models
let cachedModels = HuggingFaceHub.listCached()
print("Cached models:", cachedModels)
```

### Invalid Input Shape

```swift
// Ensure image is properly preprocessed
let preprocessor = ImagePreprocessor(imageSize: 224)

// This will fail:
// let wrongSize = MLXRandom.normal([1, 3, 112, 112])
// aim.encode(pixels: wrongSize)  // ❌ Error

// This will work:
let correctSize = MLXRandom.normal([1, 3, 224, 224])
aim.encode(pixels: correctSize)  // ✅ OK
```

## Best Practices

1. **Always use ImagePreprocessor** for consistent normalization
2. **Enable eval mode** after loading weights: `MLX.eval(model)`
3. **Use batch processing** for better performance
4. **Cache loaded models** to avoid reloading
5. **Use smaller image sizes** (224) for faster inference when accuracy allows
6. **Check model.info** to verify configuration

## Next Steps

- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for implementation details
- Check [CHANGELOG.md](CHANGELOG.md) for version history
- Read [CLAUDE.md](CLAUDE.md) for development guide
