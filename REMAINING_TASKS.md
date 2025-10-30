# Remaining Tasks

Outstanding tasks and future enhancements for swift-aim project.

## Current Status: v0.3.0 ✅

All critical issues resolved. Core functionality complete and production-ready.

---

## 🔴 High Priority (Testing)

### 1. ImagePreprocessor Test Suite
**Status**: Not started
**Estimated Effort**: 2-3 hours
**Description**: Create comprehensive tests for image preprocessing

**Test Cases Needed:**
- CGImage to tensor conversion
- Resize and center crop logic
- Normalization correctness
- Batch preprocessing
- Edge cases (small images, non-square images)
- Alpha channel handling

**Files to Create:**
```
Tests/SwiftAIMTests/ImagePreprocessorTests.swift
```

**Example Test:**
```swift
@Test("Image normalization with ImageNet stats")
func testImageNormalization() {
    let preprocessor = ImagePreprocessor(imageSize: 224)
    // Test white pixel: (255, 255, 255)
    // Expected after normalization: ((1.0 - 0.485) / 0.229, ...)
    // Verify calculations
}
```

---

### 2. Integration Tests with Real Models
**Status**: Not started
**Estimated Effort**: 4-5 hours
**Description**: Test with actual HuggingFace model weights

**Requirements:**
- Download a small AIMv2 model (e.g., base variant)
- Test complete load → inference pipeline
- Verify output shapes and values
- Test different image sizes (224, 336, 448)

**Files to Create:**
```
Tests/SwiftAIMTests/IntegrationTests.swift
```

**Test Cases:**
```swift
@Test("Load and run inference on real model")
func testRealModelInference() throws {
    // Skip if model not available
    guard FileManager.default.fileExists(atPath: modelPath) else {
        return
    }

    let aim = try AIMv2.load(fromPath: modelPath)
    // Test inference...
}
```

---

### 3. HuggingFaceHub Tests
**Status**: Not started
**Estimated Effort**: 2 hours
**Description**: Test Hub utilities

**Test Cases:**
- Model ID parsing
- Cache path generation
- URL construction
- isCached() logic
- listCached() functionality

**Files to Create:**
```
Tests/SwiftAIMTests/HuggingFaceHubTests.swift
```

---

## 🟡 Medium Priority (Improvements)

### 4. Weight Key Naming Verification
**Status**: Not started
**Estimated Effort**: 1-2 hours
**Description**: Test weight sanitization with real HuggingFace models

**Action Items:**
- Download real model safetensors
- Inspect actual key names (snake_case vs camelCase)
- Verify sanitization handles all variants
- Update sanitization if needed

**Verification:**
```swift
// Test with real weights
let weights = try SafetensorsLoader.load(from: realModelPath)
let keys = Array(weights.keys).sorted()
print("Actual HuggingFace keys:", keys)

// Verify sanitization works
let sanitized = AIMv2Model.sanitize(weights: weights)
// Check all expected keys are present
```

---

### 5. Silent Failure Error Handling
**Status**: Partial fix
**Estimated Effort**: 2 hours
**Description**: Replace silent failures with proper errors

**Files to Update:**
```
Sources/SwiftAIM/Processing/ImagePreprocessor.swift:86, 94, 102, 129
```

**Changes:**
```swift
// Current: Returns original/zero on failure
guard let context = CGContext(...) else {
    return image  // Silent failure
}

// Better: Return nil or throw
guard let context = CGContext(...) else {
    return nil  // Or throw PreprocessingError.contextCreationFailed
}
```

---

### 6. Batch Processing Optimization
**Status**: Not started
**Estimated Effort**: 2-3 hours
**Description**: Optimize batch preprocessing for memory efficiency

**Current Issue:**
```swift
// Creates N intermediate tensors
let preprocessed = images.map { preprocessor.preprocess($0) }
let batch = MLX.concatenated(preprocessed, axis: 0)
```

**Optimization:**
```swift
// Preallocate batch tensor
var batch = MLXArray.zeros([images.count, 3, imageSize, imageSize])
for (i, image) in images.enumerated() {
    let processed = preprocessor.preprocess(image)
    batch[i] = processed[0]  // Remove batch dimension
}
```

---

## 🟢 Low Priority (Enhancements)

### 7. Resize Interpolation Precision
**Status**: Not started
**Estimated Effort**: 30 minutes
**Description**: Use CGFloat instead of Float for better precision

**File**: `Sources/SwiftAIM/Processing/ImagePreprocessor.swift:68-71`

```swift
// Current
let scale = Float(targetSize) / Float(min(width, height))
let newWidth = Int(Float(width) * scale)

// Better
let scale = CGFloat(targetSize) / CGFloat(min(width, height))
let newWidth = Int(CGFloat(width) * scale)
```

---

### 8. Precompute Sincos Cache
**Status**: Not started
**Estimated Effort**: 1 hour
**Description**: Precompute sincos position embeddings

**File**: `Sources/SwiftAIM/Core/AIMv2Model.swift:18-27`

**Change:**
```swift
// Current: Lazy computation
private lazy var sinCosCache: MLXArray? = { ... }()

// Better: Eager computation in init
public init(config: AIMv2Configuration) {
    // ... existing init code ...

    // Precompute if using sincos
    if config.positionEmbeddingType == .sincos {
        self.sinCosCache = createSinCosPositionalEmbedding(...)
    }
}
```

---

### 9. Missing Resize Implementation
**Status**: Documentation needed
**Estimated Effort**: 30 minutes
**Description**: Document resize limitation or implement

**File**: `Sources/SwiftAIM/Processing/ImagePreprocessor.swift:183-187`

**Option 1 - Document:**
```swift
/// Preprocess from raw pixel data [H, W, C] in range [0, 1]
/// - Parameter pixels: Raw pixel tensor [H, W, 3] with values in [0, 1]
///                     **Must already be resized to imageSize**
/// - Returns: Preprocessed tensor [1, 3, H, W]
/// - Note: This method does not perform resizing. Use preprocess(_: CGImage)
///         for automatic resizing, or manually resize before calling this method.
```

**Option 2 - Implement:**
```swift
if H != imageSize || W != imageSize {
    // Implement simple nearest-neighbor or bilinear resize
    pixels = resizeMLXArray(pixels, to: imageSize)
}
```

---

## 🚀 Future Enhancements

### 10. Automatic Model Downloading
**Status**: Not started
**Estimated Effort**: 1-2 days
**Description**: Implement automatic download from HuggingFace Hub

**Requirements:**
- URLSession-based downloader
- Progress tracking
- Resume support
- Proper error handling

---

### 11. Performance Benchmarking
**Status**: Not started
**Estimated Effort**: 1 day
**Description**: Create comprehensive performance benchmarks

**Metrics to Track:**
- Inference time per image size
- Memory usage
- Batch throughput
- Cold start vs warm inference

---

### 12. Example Applications
**Status**: Not started
**Estimated Effort**: 2-3 days
**Description**: Create example apps

**Examples:**
1. **Image Similarity Search**
   - UI for image upload
   - Feature extraction
   - Similarity ranking

2. **Image Classification**
   - Load classifier head
   - Real-time inference
   - Top-k predictions

3. **Batch Feature Extraction**
   - Command-line tool
   - Process image directories
   - Export features to file

---

### 13. Metal Performance Shaders
**Status**: Research needed
**Estimated Effort**: 1 week
**Description**: Optimize with Metal shaders

**Areas:**
- Custom preprocessing shaders
- Optimized attention kernels
- Batch processing

---

### 14. Quantization Support
**Status**: Not started
**Estimated Effort**: 1-2 weeks
**Description**: Add int8/int4 quantization

**Benefits:**
- Reduced model size
- Faster inference
- Lower memory usage

---

## Priority Order

1. **Immediate** (Before v1.0 release)
   - [ ] ImagePreprocessor tests
   - [ ] Integration tests
   - [ ] Weight key verification

2. **Short-term** (v1.1)
   - [ ] Silent failure fixes
   - [ ] Batch optimization
   - [ ] HuggingFaceHub tests

3. **Medium-term** (v1.2+)
   - [ ] Automatic downloads
   - [ ] Performance benchmarks
   - [ ] Example applications

4. **Long-term** (v2.0+)
   - [ ] Metal optimization
   - [ ] Quantization
   - [ ] Advanced features

---

## Test Coverage Goal

**Current**: ~40% (core model and config)
**Target**: >80% for v1.0

**Missing Coverage:**
- ImagePreprocessor: 0%
- SafetensorsLoader: 0%
- AIMv2 (high-level API): 0%
- HuggingFaceHub: 0%

---

## How to Contribute

1. Pick a task from above
2. Create a feature branch: `git checkout -b feature/task-name`
3. Implement with tests
4. Run `swift test` to verify
5. Update this file with status
6. Submit PR

---

## Questions / Clarifications Needed

- [ ] Verify actual HuggingFace model key naming conventions
- [ ] Confirm expected behavior for images with transparency
- [ ] Determine if Metal optimization is needed
- [ ] Decide on quantization priority

---

**Last Updated**: 2025-10-30
**Version**: v0.3.0
**Status**: Core Complete, Testing & Enhancements Remaining
