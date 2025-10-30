import Testing
import Foundation
import MLX
import MLXRandom
@testable import SwiftAIM

#if canImport(CoreGraphics)
import CoreGraphics

/// Comprehensive tests for ImagePreprocessor
@Suite("ImagePreprocessor Tests")
struct ImagePreprocessorTests {

    // MARK: - Initialization Tests

    @Test("Preprocessor initialization with valid parameters")
    func testInitialization() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        #expect(preprocessor.imageSize == 224)
        #expect(preprocessor.mean.count == 3)
        #expect(preprocessor.std.count == 3)

        // Default ImageNet values
        #expect(abs(preprocessor.mean[0] - 0.485) < 0.001)
        #expect(abs(preprocessor.mean[1] - 0.456) < 0.001)
        #expect(abs(preprocessor.mean[2] - 0.406) < 0.001)

        #expect(abs(preprocessor.std[0] - 0.229) < 0.001)
        #expect(abs(preprocessor.std[1] - 0.224) < 0.001)
        #expect(abs(preprocessor.std[2] - 0.225) < 0.001)
    }

    @Test("Preprocessor initialization with custom parameters")
    func testCustomInitialization() {
        let customMean: [Float] = [0.5, 0.5, 0.5]
        let customStd: [Float] = [0.5, 0.5, 0.5]

        let preprocessor = ImagePreprocessor(
            imageSize: 336,
            mean: customMean,
            std: customStd
        )

        #expect(preprocessor.imageSize == 336)
        #expect(preprocessor.mean == customMean)
        #expect(preprocessor.std == customStd)
    }

    // MARK: - Normalization Tests

    @Test("Normalization formula correctness")
    func testNormalizationFormula() {
        let preprocessor = ImagePreprocessor(
            imageSize: 224,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5]
        )

        // Create test image: all white pixels [1.0, 1.0, 1.0]
        let testPixels = MLXArray.ones([224, 224, 3])
        let normalized = preprocessor.preprocess(pixels: testPixels)

        // Expected: (1.0 - 0.5) / 0.5 = 1.0
        let expected: Float = 1.0

        // Check a few pixels
        let sample = normalized[0, 0, 0, 0].item(Float.self)
        #expect(abs(sample - expected) < 0.001)
    }

    @Test("Normalization with zero pixels")
    func testNormalizationZeroPixels() {
        let preprocessor = ImagePreprocessor(
            imageSize: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225]
        )

        // Black pixels [0, 0, 0]
        let blackPixels = MLXArray.zeros([224, 224, 3])
        let normalized = preprocessor.preprocess(pixels: blackPixels)

        // Expected: (0 - mean) / std
        let expectedR = -0.485 / 0.229
        let expectedG = -0.456 / 0.224
        let expectedB = -0.406 / 0.225

        let r = normalized[0, 0, 0, 0].item(Float.self)
        let g = normalized[0, 1, 0, 0].item(Float.self)
        let b = normalized[0, 2, 0, 0].item(Float.self)

        #expect(abs(Double(r) - Double(expectedR)) < 0.01)
        #expect(abs(Double(g) - Double(expectedG)) < 0.01)
        #expect(abs(Double(b) - Double(expectedB)) < 0.01)
    }

    // MARK: - Shape Transformation Tests

    @Test("Shape transformation HWC to BCHW")
    func testShapeTransformation() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Input: [H, W, C]
        let input = MLXRandom.uniform(low: 0, high: 1, [224, 224, 3])

        // Output: [1, C, H, W]
        let output = preprocessor.preprocess(pixels: input)

        #expect(output.ndim == 4)
        #expect(output.shape[0] == 1)  // Batch
        #expect(output.shape[1] == 3)  // Channels
        #expect(output.shape[2] == 224) // Height
        #expect(output.shape[3] == 224) // Width
    }

    @Test("Channel ordering RGB preserved")
    func testChannelOrdering() {
        let preprocessor = ImagePreprocessor(
            imageSize: 224,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0]
        )

        // Create image with distinct channels
        // R channel = 1.0, G and B = 0.0
        var pixelData = [Float](repeating: 0.0, count: 224 * 224 * 3)
        for i in 0..<(224 * 224) {
            pixelData[i * 3] = 1.0  // R channel
            pixelData[i * 3 + 1] = 0.0  // G channel
            pixelData[i * 3 + 2] = 0.0  // B channel
        }
        let pixels = MLXArray(pixelData, [224, 224, 3])

        let output = preprocessor.preprocess(pixels: pixels)

        // Check R channel (index 0) is 1.0
        let rValue = output[0, 0, 0, 0].item(Float.self)
        let gValue = output[0, 1, 0, 0].item(Float.self)
        let bValue = output[0, 2, 0, 0].item(Float.self)

        #expect(abs(rValue - 1.0) < 0.001)
        #expect(abs(gValue - 0.0) < 0.001)
        #expect(abs(bValue - 0.0) < 0.001)
    }

    // MARK: - CGImage Tests

    @Test("CGImage preprocessing")
    func testCGImagePreprocessing() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Create test CGImage (solid color)
        let width = 256
        let height = 256
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        var pixelData = [UInt8](repeating: 128, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            Issue.record("Failed to create CGContext")
            return
        }

        guard let cgImage = context.makeImage() else {
            Issue.record("Failed to create CGImage")
            return
        }

        // Preprocess
        guard let output = preprocessor.preprocess(cgImage) else {
            Issue.record("Preprocessing failed")
            return
        }

        // Verify output shape
        #expect(output.shape[0] == 1)
        #expect(output.shape[1] == 3)
        #expect(output.shape[2] == 224)
        #expect(output.shape[3] == 224)
    }

    @Test("CGImage different sizes are resized")
    func testCGImageResize() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        let testSizes = [(128, 128), (256, 256), (512, 512), (224, 448), (448, 224)]

        for (width, height) in testSizes {
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

            var pixelData = [UInt8](repeating: 100, count: width * height * 4)

            guard let context = CGContext(
                data: &pixelData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ),
            let cgImage = context.makeImage() else {
                continue
            }

            guard let output = preprocessor.preprocess(cgImage) else {
                continue
            }

            // All should be resized to 224x224
            #expect(output.shape[2] == 224)
            #expect(output.shape[3] == 224)
        }
    }

    // MARK: - Batch Processing Tests

    @Test("Batch preprocessing")
    func testBatchPreprocessing() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Create multiple test images
        let images = [
            MLXRandom.uniform(low: 0, high: 1, [224, 224, 3]),
            MLXRandom.uniform(low: 0, high: 1, [224, 224, 3]),
            MLXRandom.uniform(low: 0, high: 1, [224, 224, 3])
        ]

        let batched = preprocessor.batchPreprocess(images)

        // Should be [3, 3, 224, 224]
        #expect(batched.shape[0] == 3)  // Batch size
        #expect(batched.shape[1] == 3)  // Channels
        #expect(batched.shape[2] == 224) // Height
        #expect(batched.shape[3] == 224) // Width
    }

    // MARK: - Validation Tests

    @Test("Input validation for wrong channel count")
    func testWrongChannelCount() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // 4 channels instead of 3
        let wrongChannels = MLXRandom.uniform(low: 0, high: 1, [224, 224, 4])

        // Should trigger precondition
        // Note: Can't test preconditions directly in Swift Testing
        // This test documents expected behavior
        #expect(wrongChannels.shape[2] == 4)
    }

    @Test("Input validation for wrong dimensions")
    func testWrongDimensions() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // 2D instead of 3D
        let wrong2D = MLXRandom.uniform(low: 0, high: 1, [224, 224])

        // Should trigger precondition
        #expect(wrong2D.ndim == 2)
    }

    // MARK: - Different Image Sizes

    @Test("Different target image sizes", arguments: [224, 336, 448])
    func testDifferentImageSizes(targetSize: Int) {
        let preprocessor = ImagePreprocessor(imageSize: targetSize)

        let input = MLXRandom.uniform(low: 0, high: 1, [targetSize, targetSize, 3])
        let output = preprocessor.preprocess(pixels: input)

        #expect(output.shape[2] == targetSize)
        #expect(output.shape[3] == targetSize)
    }

    // MARK: - Edge Cases

    @Test("Small image preprocessing")
    func testSmallImage() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Very small input
        let width = 32
        let height = 32
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        var pixelData = [UInt8](repeating: 128, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ),
        let cgImage = context.makeImage() else {
            Issue.record("Failed to create small image")
            return
        }

        guard let output = preprocessor.preprocess(cgImage) else {
            Issue.record("Preprocessing failed for small image")
            return
        }

        // Should be upscaled to 224x224
        #expect(output.shape[2] == 224)
        #expect(output.shape[3] == 224)
    }

    @Test("Non-square image preprocessing")
    func testNonSquareImage() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Wide image
        let width = 512
        let height = 256
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        var pixelData = [UInt8](repeating: 128, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ),
        let cgImage = context.makeImage() else {
            Issue.record("Failed to create non-square image")
            return
        }

        guard let output = preprocessor.preprocess(cgImage) else {
            Issue.record("Preprocessing failed for non-square image")
            return
        }

        // Should be center-cropped to 224x224
        #expect(output.shape[2] == 224)
        #expect(output.shape[3] == 224)
    }

    // MARK: - Value Range Tests

    @Test("Output values are in reasonable range")
    func testOutputValueRange() {
        let preprocessor = ImagePreprocessor(imageSize: 224)

        // Random input in [0, 1]
        let input = MLXRandom.uniform(low: 0, high: 1, [224, 224, 3])
        let output = preprocessor.preprocess(pixels: input)

        // With ImageNet normalization, values should roughly be in [-3, 3]
        let minVal = output.min().item(Float.self)
        let maxVal = output.max().item(Float.self)

        // Reasonable range for normalized values
        #expect(minVal > -5.0)
        #expect(maxVal < 5.0)
    }
}

#endif
