import Testing
import Foundation
@testable import SwiftAIM

/// Comprehensive tests for HuggingFace Hub utilities
@Suite("HuggingFaceHub Tests")
struct HuggingFaceHubTests {

    // MARK: - ModelRepo Initialization Tests

    @Test("ModelRepo initialization from valid model ID")
    func testModelRepoValidInit() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large-patch14-224")

        #expect(repo.owner == "apple")
        #expect(repo.name == "aimv2-large-patch14-224")
        #expect(repo.fullName == "apple/aimv2-large-patch14-224")
    }

    @Test("ModelRepo initialization with different model IDs", arguments: [
        ("apple/aimv2-base", "apple", "aimv2-base"),
        ("huggingface/model-name", "huggingface", "model-name"),
        ("org/very-long-model-name-123", "org", "very-long-model-name-123")
    ])
    func testModelRepoVariousIDs(args: (String, String, String)) throws {
        let (modelID, expectedOwner, expectedName) = args
        let repo = try HuggingFaceHub.ModelRepo(modelID: modelID)

        #expect(repo.owner == expectedOwner)
        #expect(repo.name == expectedName)
    }

    @Test("ModelRepo initialization fails with invalid model ID - no slash")
    func testModelRepoInvalidNoSlash() {
        #expect(throws: HubError.self) {
            _ = try HuggingFaceHub.ModelRepo(modelID: "invalidmodelid")
        }
    }

    @Test("ModelRepo initialization fails with invalid model ID - multiple slashes")
    func testModelRepoInvalidMultipleSlashes() {
        #expect(throws: HubError.self) {
            _ = try HuggingFaceHub.ModelRepo(modelID: "owner/name/extra")
        }
    }

    @Test("ModelRepo initialization fails with invalid model ID - empty string")
    func testModelRepoInvalidEmpty() {
        #expect(throws: HubError.self) {
            _ = try HuggingFaceHub.ModelRepo(modelID: "")
        }
    }

    @Test("ModelRepo initialization fails with invalid model ID - only slash")
    func testModelRepoInvalidOnlySlash() {
        #expect(throws: HubError.self) {
            _ = try HuggingFaceHub.ModelRepo(modelID: "/")
        }
    }

    @Test("ModelRepo direct initialization")
    func testModelRepoDirectInit() {
        let repo = HuggingFaceHub.ModelRepo(owner: "apple", name: "aimv2-large")

        #expect(repo.owner == "apple")
        #expect(repo.name == "aimv2-large")
        #expect(repo.fullName == "apple/aimv2-large")
    }

    // MARK: - URL Construction Tests

    @Test("ModelRepo URL construction")
    func testModelRepoURL() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")

        guard let url = repo.url else {
            Issue.record("URL should not be nil")
            return
        }

        #expect(url.absoluteString == "https://huggingface.co/apple/aimv2-large")
    }

    @Test("ModelRepo files URL construction")
    func testModelRepoFilesURL() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")

        guard let url = repo.filesURL else {
            Issue.record("Files URL should not be nil")
            return
        }

        #expect(url.absoluteString == "https://huggingface.co/apple/aimv2-large/tree/main")
    }

    // MARK: - Download URL Tests

    @Test("Download URL generation for config.json")
    func testDownloadURLConfig() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")
        let url = try HuggingFaceHub.downloadURL(for: repo, filename: "config.json")

        let expected = "https://huggingface.co/apple/aimv2-large/resolve/main/config.json"
        #expect(url.absoluteString == expected)
    }

    @Test("Download URL generation for model.safetensors")
    func testDownloadURLSafetensors() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")
        let url = try HuggingFaceHub.downloadURL(for: repo, filename: "model.safetensors")

        let expected = "https://huggingface.co/apple/aimv2-large/resolve/main/model.safetensors"
        #expect(url.absoluteString == expected)
    }

    @Test("Download URL generation with custom revision")
    func testDownloadURLCustomRevision() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")
        let url = try HuggingFaceHub.downloadURL(for: repo, filename: "config.json", revision: "v1.0")

        let expected = "https://huggingface.co/apple/aimv2-large/resolve/v1.0/config.json"
        #expect(url.absoluteString == expected)
    }

    @Test("Download URL with subdirectory path")
    func testDownloadURLSubdirectory() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")
        let url = try HuggingFaceHub.downloadURL(for: repo, filename: "weights/model.safetensors")

        let expected = "https://huggingface.co/apple/aimv2-large/resolve/main/weights/model.safetensors"
        #expect(url.absoluteString == expected)
    }

    // MARK: - Cache Path Tests

    @Test("Cache path generation")
    func testCachePath() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "apple/aimv2-large")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)

        let expectedSuffix = ".cache/huggingface/hub/apple/aimv2-large"
        #expect(cachePath.path.hasSuffix(expectedSuffix))
    }

    @Test("Cache path contains owner and model name")
    func testCachePathStructure() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "test-owner/test-model")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)

        #expect(cachePath.path.contains("test-owner"))
        #expect(cachePath.path.contains("test-model"))
    }

    @Test("Cache path is under home directory")
    func testCachePathLocation() {
        let cacheDir = HuggingFaceHub.cacheDirectory
        let homeDir = FileManager.default.homeDirectoryForCurrentUser

        #expect(cacheDir.path.hasPrefix(homeDir.path))
    }

    // MARK: - Cache Check Tests

    @Test("isCached returns false for non-existent model")
    func testIsCachedNonExistent() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "nonexistent/model-12345")
        let isCached = HuggingFaceHub.isCached(repo)

        #expect(!isCached)
    }

    @Test("isCached checks for default required files")
    func testIsCachedDefaultFiles() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "test/model")

        // Create temporary cache directory
        let cachePath = HuggingFaceHub.localCachePath(for: repo)
        try? FileManager.default.createDirectory(at: cachePath, withIntermediateDirectories: true)

        // Without files, should not be cached
        #expect(!HuggingFaceHub.isCached(repo))

        // Create config.json
        let configPath = cachePath.appendingPathComponent("config.json")
        try "{}".write(to: configPath, atomically: true, encoding: .utf8)

        // Still missing model.safetensors
        #expect(!HuggingFaceHub.isCached(repo))

        // Create model.safetensors
        let weightsPath = cachePath.appendingPathComponent("model.safetensors")
        try Data().write(to: weightsPath)

        // Now should be cached
        #expect(HuggingFaceHub.isCached(repo))

        // Cleanup
        try? FileManager.default.removeItem(at: cachePath)
    }

    @Test("isCached with custom required files")
    func testIsCachedCustomFiles() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "test/custom-model")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)

        try? FileManager.default.createDirectory(at: cachePath, withIntermediateDirectories: true)

        // Check for custom file
        #expect(!HuggingFaceHub.isCached(repo, requiredFiles: ["custom.txt"]))

        // Create custom file
        let customPath = cachePath.appendingPathComponent("custom.txt")
        try "content".write(to: customPath, atomically: true, encoding: .utf8)

        // Should be cached now
        #expect(HuggingFaceHub.isCached(repo, requiredFiles: ["custom.txt"]))

        // Cleanup
        try? FileManager.default.removeItem(at: cachePath)
    }

    // MARK: - List Cached Tests

    @Test("listCached returns array")
    func testListCachedReturnsArray() {
        // Should return an array (may be empty or contain models)
        let cached = HuggingFaceHub.listCached()

        // Verify it's an array and each element has config.json
        for modelPath in cached {
            let configPath = modelPath.appendingPathComponent("config.json")
            #expect(FileManager.default.fileExists(atPath: configPath.path))
        }
    }

    @Test("listCached finds models with config.json")
    func testListCachedWithModel() throws {
        // Create temporary model cache
        let testRepo = try HuggingFaceHub.ModelRepo(modelID: "test-list/model-123")
        let cachePath = HuggingFaceHub.localCachePath(for: testRepo)

        try? FileManager.default.createDirectory(at: cachePath, withIntermediateDirectories: true)

        // Create config.json
        let configPath = cachePath.appendingPathComponent("config.json")
        try "{}".write(to: configPath, atomically: true, encoding: .utf8)

        // List should include this model
        let cached = HuggingFaceHub.listCached()
        let containsTest = cached.contains { $0.path.contains("test-list") && $0.path.contains("model-123") }

        #expect(containsTest)

        // Cleanup
        try? FileManager.default.removeItem(at: cachePath)
    }

    // MARK: - Error Handling Tests

    @Test("HubError invalidModelID description")
    func testHubErrorInvalidModelID() {
        let error = HubError.invalidModelID("test-id")
        let description = error.description

        #expect(description.contains("Invalid model ID"))
        #expect(description.contains("test-id"))
    }

    @Test("HubError downloadFailed description")
    func testHubErrorDownloadFailed() {
        let error = HubError.downloadFailed("network error")
        let description = error.description

        #expect(description.contains("Download failed"))
        #expect(description.contains("network error"))
    }

    @Test("HubError modelNotCached description")
    func testHubErrorModelNotCached() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "test/model")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)
        let instructions = "test instructions"

        let error = HubError.modelNotCached(
            modelID: "test/model",
            cachePath: cachePath,
            downloadInstructions: instructions
        )
        let description = error.description

        #expect(description.contains("test/model"))
        #expect(description.contains("not cached"))
    }

    // MARK: - Integration Tests

    @Test("Full workflow: parse ID, generate URLs, check cache")
    func testFullWorkflow() throws {
        let modelID = "apple/aimv2-base-patch14-224"

        // Parse model ID
        let repo = try HuggingFaceHub.ModelRepo(modelID: modelID)
        #expect(repo.fullName == modelID)

        // Generate URLs
        let configURL = try HuggingFaceHub.downloadURL(for: repo, filename: "config.json")
        #expect(configURL.absoluteString.contains("apple/aimv2-base-patch14-224"))

        let weightsURL = try HuggingFaceHub.downloadURL(for: repo, filename: "model.safetensors")
        #expect(weightsURL.absoluteString.contains("model.safetensors"))

        // Check cache path
        let cachePath = HuggingFaceHub.localCachePath(for: repo)
        #expect(cachePath.path.contains("apple"))
        #expect(cachePath.path.contains("aimv2-base-patch14-224"))

        // Check if cached functionality works (returns true or false)
        let isCached = HuggingFaceHub.isCached(repo)
        // Should be false for non-downloaded model, or true if actually cached
        // Just verify the function runs without error
        _ = isCached
    }

    @Test("Error handling: invalid ID throws before URL generation")
    func testErrorHandlingOrder() {
        // Invalid model ID should fail at parsing, not URL generation
        #expect(throws: HubError.self) {
            let repo = try HuggingFaceHub.ModelRepo(modelID: "invalid-id-no-slash")
            _ = try HuggingFaceHub.downloadURL(for: repo, filename: "config.json")
        }
    }

    // MARK: - Edge Cases

    @Test("ModelRepo with special characters in name")
    func testModelRepoSpecialCharacters() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "owner/model-name_v2.0")

        #expect(repo.name == "model-name_v2.0")
        #expect(repo.fullName == "owner/model-name_v2.0")
    }

    @Test("Download URL with special characters")
    func testDownloadURLSpecialCharacters() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "owner/model-v2.0")
        let url = try HuggingFaceHub.downloadURL(for: repo, filename: "weights/part-1.safetensors")

        #expect(url.absoluteString.contains("model-v2.0"))
        #expect(url.absoluteString.contains("part-1.safetensors"))
    }

    @Test("Cache path for very long model name")
    func testCachePathLongName() throws {
        let longName = "very-long-model-name-with-many-words-and-numbers-123456789"
        let repo = try HuggingFaceHub.ModelRepo(modelID: "owner/\(longName)")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)

        #expect(cachePath.path.contains(longName))
    }

    @Test("Multiple required files check")
    func testMultipleRequiredFiles() throws {
        let repo = try HuggingFaceHub.ModelRepo(modelID: "test/multi-files")
        let cachePath = HuggingFaceHub.localCachePath(for: repo)

        try? FileManager.default.createDirectory(at: cachePath, withIntermediateDirectories: true)

        let requiredFiles = ["config.json", "model.safetensors", "tokenizer.json"]

        // None exist
        #expect(!HuggingFaceHub.isCached(repo, requiredFiles: requiredFiles))

        // Create first two files
        try "{}".write(to: cachePath.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: cachePath.appendingPathComponent("model.safetensors"))

        // Still missing one
        #expect(!HuggingFaceHub.isCached(repo, requiredFiles: requiredFiles))

        // Create third file
        try "{}".write(to: cachePath.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)

        // All present now
        #expect(HuggingFaceHub.isCached(repo, requiredFiles: requiredFiles))

        // Cleanup
        try? FileManager.default.removeItem(at: cachePath)
    }
}
