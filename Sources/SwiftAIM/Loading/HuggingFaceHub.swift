import Foundation

/// HuggingFace Hub integration for model loading
///
/// Provides utilities for downloading and caching models from HuggingFace Hub.
///
/// **Note**: This is a basic implementation. Full Hub integration requires
/// additional dependencies for HTTP downloads and progress tracking.
public struct HuggingFaceHub {

    /// HuggingFace Hub base URL
    public static let baseURL = "https://huggingface.co"

    /// Default cache directory
    public static var cacheDirectory: URL {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".cache/huggingface/hub")
    }

    /// Model repository information
    public struct ModelRepo {
        public let owner: String
        public let name: String

        public init(owner: String, name: String) {
            self.owner = owner
            self.name = name
        }

        /// Create from model ID (e.g., "apple/aimv2-large-patch14-224")
        public init(modelID: String) throws {
            let components = modelID.split(separator: "/")
            guard components.count == 2 else {
                throw HubError.invalidModelID("Model ID must be in format 'owner/name', got: '\(modelID)'")
            }
            self.owner = String(components[0])
            self.name = String(components[1])
        }

        public var fullName: String {
            "\(owner)/\(name)"
        }

        public var url: URL? {
            URL(string: "\(HuggingFaceHub.baseURL)/\(owner)/\(name)")
        }

        public var filesURL: URL? {
            URL(string: "\(HuggingFaceHub.baseURL)/\(owner)/\(name)/tree/main")
        }
    }

    /// Get local cache path for a model
    /// - Parameter repo: Model repository
    /// - Returns: Local cache directory URL
    public static func localCachePath(for repo: ModelRepo) -> URL {
        return cacheDirectory
            .appendingPathComponent(repo.owner)
            .appendingPathComponent(repo.name)
    }

    /// Check if model is cached locally
    /// - Parameters:
    ///   - repo: Model repository
    ///   - requiredFiles: Files that must exist (default: config.json and model.safetensors)
    /// - Returns: True if all required files exist
    public static func isCached(
        _ repo: ModelRepo,
        requiredFiles: [String] = ["config.json", "model.safetensors"]
    ) -> Bool {
        let cachePath = localCachePath(for: repo)

        for file in requiredFiles {
            let filePath = cachePath.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                return false
            }
        }

        return true
    }

    /// Get download URL for a file in the repository
    /// - Parameters:
    ///   - repo: Model repository
    ///   - filename: File name
    ///   - revision: Git revision (default: "main")
    /// - Returns: Download URL
    /// - Throws: HubError if URL construction fails
    public static func downloadURL(
        for repo: ModelRepo,
        filename: String,
        revision: String = "main"
    ) throws -> URL {
        guard let url = URL(string: "\(baseURL)/\(repo.fullName)/resolve/\(revision)/\(filename)") else {
            throw HubError.invalidModelID("Invalid URL for model: \(repo.fullName), file: \(filename)")
        }
        return url
    }

    /// Load model from Hub (if cached) or provide download instructions
    /// - Parameter modelID: Model ID (e.g., "apple/aimv2-large-patch14-224")
    /// - Returns: Loaded AIMv2 instance
    /// - Throws: HubError if model is not cached or modelID is invalid
    public static func load(modelID: String) throws -> AIMv2 {
        let repo = try ModelRepo(modelID: modelID)
        let cachePath = localCachePath(for: repo)

        guard isCached(repo) else {
            throw HubError.modelNotCached(
                modelID: modelID,
                cachePath: cachePath,
                downloadInstructions: generateDownloadInstructions(for: repo)
            )
        }

        return try AIMv2.load(from: cachePath)
    }

    /// Generate download instructions for a model
    private static func generateDownloadInstructions(for repo: ModelRepo) -> String {
        let cachePath = localCachePath(for: repo).path
        let configURL = (try? downloadURL(for: repo, filename: "config.json"))?.absoluteString ?? "Unable to generate URL"
        let weightsURL = (try? downloadURL(for: repo, filename: "model.safetensors"))?.absoluteString ?? "Unable to generate URL"
        let repoURL = repo.url?.absoluteString ?? HuggingFaceHub.baseURL

        return """

        Model not found in cache. To download:

        1. Create cache directory:
           mkdir -p "\(cachePath)"

        2. Download config.json:
           curl -L "\(configURL)" -o "\(cachePath)/config.json"

        3. Download model.safetensors:
           curl -L "\(weightsURL)" -o "\(cachePath)/model.safetensors"

        Or visit: \(repoURL)
        """
    }

    /// List locally cached models
    /// - Returns: Array of cached model paths
    public static func listCached() -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: cacheDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var models: [URL] = []

        for case let url as URL in enumerator {
            let configPath = url.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: configPath.path) {
                models.append(url)
            }
        }

        return models
    }
}

/// HuggingFace Hub errors
public enum HubError: Error, CustomStringConvertible {
    case modelNotCached(modelID: String, cachePath: URL, downloadInstructions: String)
    case downloadFailed(String)
    case invalidModelID(String)

    public var description: String {
        switch self {
        case .modelNotCached(let modelID, _, let instructions):
            return "Model '\(modelID)' not cached.\(instructions)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .invalidModelID(let id):
            return "Invalid model ID: \(id)"
        }
    }
}
