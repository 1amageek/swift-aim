import MLX
import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
#endif

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// Preprocessing errors
public enum PreprocessingError: Error, CustomStringConvertible {
    case contextCreationFailed
    case imageCreationFailed
    case croppingFailed

    public var description: String {
        switch self {
        case .contextCreationFailed:
            return "Failed to create CGContext for image processing"
        case .imageCreationFailed:
            return "Failed to create CGImage from context"
        case .croppingFailed:
            return "Failed to crop image to target size"
        }
    }
}

/// Image preprocessing for AIMv2 models
///
/// Handles image loading, resizing, and normalization using ImageNet statistics.
public class ImagePreprocessor {

    /// Target image size (height and width)
    public let imageSize: Int

    /// ImageNet mean values for normalization [R, G, B]
    public let mean: [Float]

    /// ImageNet standard deviation for normalization [R, G, B]
    public let std: [Float]

    /// Initializer
    /// - Parameters:
    ///   - imageSize: Target image size (224, 336, or 448)
    ///   - mean: Mean values for normalization (default: ImageNet mean)
    ///   - std: Standard deviation for normalization (default: ImageNet std)
    public init(
        imageSize: Int = 224,
        mean: [Float] = [0.485, 0.456, 0.406],
        std: [Float] = [0.229, 0.224, 0.225]
    ) {
        precondition(imageSize > 0, "imageSize must be positive")
        precondition(mean.count == 3, "mean must have 3 values (RGB)")
        precondition(std.count == 3, "std must have 3 values (RGB)")

        self.imageSize = imageSize
        self.mean = mean
        self.std = std
    }

    #if canImport(CoreGraphics)

    /// Preprocess image from CGImage
    /// - Parameter image: Input CGImage
    /// - Returns: Preprocessed tensor [1, 3, H, W] ready for model input, or nil if preprocessing fails
    public func preprocess(_ image: CGImage) -> MLXArray? {
        do {
            // Resize and crop
            let resized = try resizeAndCrop(image: image, targetSize: imageSize)

            // Convert to MLXArray [H, W, C]
            let pixels = try cgImageToMLXArray(resized)

            // Normalize and transpose to [1, C, H, W]
            return normalizeAndTranspose(pixels)
        } catch {
            // Log error for debugging (in production, you might want proper logging)
            print("Image preprocessing failed: \(error)")
            return nil
        }
    }

    /// Resize and center crop CGImage
    private func resizeAndCrop(image: CGImage, targetSize: Int) throws -> CGImage {
        let width = image.width
        let height = image.height

        // Calculate scale to make the shorter side equal to targetSize
        let scale = CGFloat(targetSize) / CGFloat(min(width, height))
        let newWidth = Int(CGFloat(width) * scale)
        let newHeight = Int(CGFloat(height) * scale)

        // Create resized bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: newWidth,
            height: newHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PreprocessingError.contextCreationFailed
        }

        // Draw resized image
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))

        guard let resizedImage = context.makeImage() else {
            throw PreprocessingError.imageCreationFailed
        }

        // Center crop
        let cropX = (newWidth - targetSize) / 2
        let cropY = (newHeight - targetSize) / 2
        let cropRect = CGRect(x: cropX, y: cropY, width: targetSize, height: targetSize)

        guard let croppedImage = resizedImage.cropping(to: cropRect) else {
            throw PreprocessingError.croppingFailed
        }

        return croppedImage
    }

    /// Batch preprocess multiple CGImages efficiently
    /// - Parameter images: Array of CGImages
    /// - Returns: Batched tensor [B, 3, H, W], or nil if any preprocessing fails
    public func batchPreprocess(_ images: [CGImage]) -> MLXArray? {
        // Preprocess all images
        var processedImages: [MLXArray] = []
        processedImages.reserveCapacity(images.count)

        for image in images {
            guard let processed = preprocess(image) else {
                return nil
            }
            processedImages.append(processed)
        }

        // Concatenate into batch
        // Note: Each processed image is [1, 3, H, W], so concatenate on axis 0
        return MLX.concatenated(processedImages, axis: 0)
    }

    /// Convert CGImage to MLXArray [H, W, C]
    private func cgImageToMLXArray(_ image: CGImage) throws -> MLXArray {
        let width = image.width
        let height = image.height

        // Create bitmap context to get pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw PreprocessingError.contextCreationFailed
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA to RGB float array
        var floatPixels = [Float](repeating: 0, count: width * height * 3)

        for i in 0..<(width * height) {
            let rgbaIndex = i * 4
            let rgbIndex = i * 3

            // Normalize to [0, 1]
            floatPixels[rgbIndex] = Float(pixelData[rgbaIndex]) / 255.0     // R
            floatPixels[rgbIndex + 1] = Float(pixelData[rgbaIndex + 1]) / 255.0  // G
            floatPixels[rgbIndex + 2] = Float(pixelData[rgbaIndex + 2]) / 255.0  // B
        }

        // Create MLXArray [H, W, C]
        return MLXArray(floatPixels, [height, width, 3])
    }

    #endif

    /// Normalize and transpose tensor from [H, W, C] to [1, C, H, W]
    private func normalizeAndTranspose(_ pixels: MLXArray) -> MLXArray {
        // Split into channels [H, W, 3] -> 3 x [H, W]
        var normalizedChannels: [MLXArray] = []

        for c in 0..<3 {
            let channel = pixels[0..., 0..., c]
            // Normalize: (pixel - mean) / std
            let normalized = (channel - mean[c]) / std[c]
            normalizedChannels.append(normalized)
        }

        // Stack channels [3, H, W]
        let stacked = MLX.stacked(normalizedChannels, axis: 0)

        // Add batch dimension [1, 3, H, W]
        return stacked.expandedDimensions(axis: 0)
    }

    /// Preprocess from raw pixel data [H, W, C] in range [0, 1]
    ///
    /// **Important**: This method does not perform resizing. Input pixels must already
    /// be resized to `imageSize x imageSize`. For automatic resizing, use
    /// `preprocess(_ image: CGImage)` instead.
    ///
    /// - Parameter pixels: Raw pixel tensor [H, W, 3] with values in [0, 1]
    ///                     **Must already be resized to imageSize**
    /// - Returns: Preprocessed tensor [1, 3, H, W]
    /// - Precondition: pixels must be [imageSize, imageSize, 3]
    ///
    /// - Note: This method only performs normalization and channel reordering.
    ///         If you need resizing, either:
    ///         1. Use `preprocess(_ image: CGImage)` which handles resizing automatically
    ///         2. Manually resize your MLXArray before calling this method
    public func preprocess(pixels: MLXArray) -> MLXArray {
        precondition(pixels.ndim == 3, "Expected 3D input [H, W, C]")
        precondition(pixels.shape[2] == 3, "Expected 3 channels (RGB)")

        // Validate size matches (no resizing performed)
        let H = pixels.shape[0]
        let W = pixels.shape[1]

        precondition(H == imageSize && W == imageSize,
                     "Input size [\(H), \(W)] must match expected [\(imageSize), \(imageSize)]. " +
                     "This method does not resize. Use preprocess(_: CGImage) for automatic resizing.")

        return normalizeAndTranspose(pixels)
    }

    /// Batch preprocess multiple images
    /// - Parameter images: Array of pixel tensors [H, W, C]
    /// - Returns: Batched tensor [B, 3, H, W]
    public func batchPreprocess(_ images: [MLXArray]) -> MLXArray {
        let processed = images.map { preprocess(pixels: $0) }
        return MLX.concatenated(processed, axis: 0)
    }
}

#if canImport(AppKit)
extension ImagePreprocessor {
    /// Preprocess image from NSImage (macOS)
    /// - Parameter image: Input NSImage
    /// - Returns: Preprocessed tensor [1, 3, H, W]
    public func preprocess(_ image: NSImage) -> MLXArray? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        return preprocess(cgImage)
    }
}
#endif

#if canImport(UIKit)
extension ImagePreprocessor {
    /// Preprocess image from UIImage (iOS)
    /// - Parameter image: Input UIImage
    /// - Returns: Preprocessed tensor [1, 3, H, W]
    public func preprocess(_ image: UIImage) -> MLXArray? {
        guard let cgImage = image.cgImage else {
            return nil
        }
        return preprocess(cgImage)
    }
}
#endif
