# Swift-AIM 設計書 v2.0

**推論専用 | Swift 6.2対応 | MLX Swift統合**

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [アーキテクチャ設計](#2-アーキテクチャ設計)
3. [コアAPI設計](#3-コアapi設計)
4. [レイヤー実装詳細](#4-レイヤー実装詳細)
5. [モデルローディング](#5-モデルローディング)
6. [画像前処理](#6-画像前処理)
7. [実装計画](#7-実装計画)
8. [使用例](#8-使用例)
9. [テスト戦略](#9-テスト戦略)

---

## 1. プロジェクト概要

### 1.1 目的

HuggingFaceの事前学習済みAIMv2（Autoregressive Image Models v2）モデルをMLX Swift上で動作させる推論専用ライブラリを作成する。

### 1.2 主要要件

- ✅ **推論のみ**（学習機能なし）
- ✅ **Swift 6.2**（strict concurrency完全対応）
- ✅ **MLX Swift 0.10.0+**
- ✅ **Apple Silicon最適化**
- ✅ **HuggingFace Hub統合**

### 1.3 非機能要件

- 型安全性とSwift Concurrency完全対応
- メモリ効率的な推論
- わかりやすいAPI設計
- ドキュメント完備

### 1.4 対応モデル

#### Phase 1（優先度高）
- `apple/aimv2-large-patch14-224` (0.3B, 224px)
- `apple/aimv2-large-patch14-336` (0.3B, 336px)

#### Phase 2
- `apple/aimv2-huge-patch14-224` (0.7B)
- `apple/aimv2-1B-patch14-224` (1B)

#### Phase 3（拡張機能）
- `apple/aimv2-large-patch14-native` (可変解像度)
- `apple/aimv2-large-patch14-224-lit` (マルチモーダル)

---

## 2. アーキテクチャ設計

### 2.1 ディレクトリ構成

```
swift-aim/
├── Package.swift
├── CLAUDE.md                          # Claude Code向けガイド
├── DESIGN.md                          # 本設計書
├── README.md                          # ユーザー向けドキュメント
│
├── Sources/
│   └── SwiftAIM/
│       ├── Core/
│       │   ├── AIMv2Model.swift          # メインビジョンエンコーダー
│       │   └── AIMv2Configuration.swift  # モデル設定構造体
│       │
│       ├── Layers/
│       │   ├── PatchEmbed.swift          # パッチ埋め込みレイヤー
│       │   ├── Attention.swift           # マルチヘッドセルフアテンション
│       │   ├── MLP.swift                 # フィードフォワードネットワーク
│       │   ├── TransformerBlock.swift    # Transformerブロック
│       │   └── PositionalEmbedding.swift # 位置埋め込み（sincos/absolute）
│       │
│       ├── Loading/
│       │   ├── ModelLoader.swift         # HuggingFaceモデルローダー
│       │   ├── WeightLoader.swift        # safetensors読み込み
│       │   └── WeightConverter.swift     # PyTorch→MLX重み変換
│       │
│       ├── Processing/
│       │   └── ImagePreprocessor.swift   # 画像前処理パイプライン
│       │
│       ├── Utils/
│       │   ├── HubAPI.swift              # HuggingFace Hub API
│       │   └── FileCache.swift           # ローカルキャッシュ管理
│       │
│       └── SwiftAIM.swift                # 公開API
│
├── Tests/
│   └── SwiftAIMTests/
│       ├── ModelTests.swift              # モデル全体のテスト
│       ├── LayerTests.swift              # 各レイヤーのテスト
│       ├── LoadingTests.swift            # ローディング機能のテスト
│       └── IntegrationTests.swift        # 統合テスト
│
└── Examples/
    └── FeatureExtraction/
        ├── main.swift                    # サンプルコード
        └── README.md
```

### 2.2 依存関係

```swift
// Package.swift
// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "swift-aim",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "SwiftAIM",
            targets: ["SwiftAIM"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
    ],
    targets: [
        .target(
            name: "SwiftAIM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "SwiftAIMTests",
            dependencies: ["SwiftAIM"]
        ),
    ]
)
```

### 2.3 アーキテクチャ図

```
┌─────────────────────────────────────────────────────────┐
│                    Public API Layer                      │
│  ┌──────────────┐         ┌──────────────────────────┐ │
│  │  SwiftAIM    │         │ AIMv2ModelLoader         │ │
│  │  (Simple)    │────────▶│ (Hub Integration)        │ │
│  └──────────────┘         └──────────────────────────┘ │
└───────────────────────────────┬─────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────┐
│                    Core Model Layer                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │           AIMv2Model (Module)                    │   │
│  │  ┌──────────────┐  ┌──────────────────────────┐ │   │
│  │  │ PatchEmbed   │  │  [TransformerBlock] x N  │ │   │
│  │  └──────────────┘  └──────────────────────────┘ │   │
│  │  ┌──────────────┐  ┌──────────────────────────┐ │   │
│  │  │ Position     │  │  LayerNorm               │ │   │
│  │  │ Embedding    │  │                          │ │   │
│  │  └──────────────┘  └──────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────┐
│                   Layer Components                       │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │ PatchEmbed    │  │  Attention    │  │    MLP      │ │
│  │  (Conv2d)     │  │  (Multi-head) │  │  (Linear)   │ │
│  └───────────────┘  └───────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────┐
│                  MLX Framework Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   MLX    │  │  MLXNN   │  │ MLXFast  │              │
│  │  (Core)  │  │ (Layers) │  │  (Ops)   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## 3. コアAPI設計

### 3.1 AIMv2Configuration

モデルの構成情報を保持する設定構造体。HuggingFaceの`config.json`から読み込まれる。

```swift
import Foundation

/// AIMv2モデルの設定
public struct AIMv2Configuration: Codable, Sendable {
    /// モデルタイプ（例: "aimv2-large-patch14-224"）
    public let modelType: String

    /// 埋め込み次元数（例: 768）
    public let hiddenSize: Int

    /// Transformerレイヤー数（例: 12）
    public let numHiddenLayers: Int

    /// アテンションヘッド数（例: 12）
    public let numAttentionHeads: Int

    /// MLP中間層次元数（例: 3072）
    public let intermediateSize: Int

    /// 入力画像サイズ（ピクセル）（例: 224, 336, 448）
    public let imageSize: Int

    /// パッチサイズ（固定14）
    public let patchSize: Int

    /// 入力チャンネル数（RGB=3）
    public let numChannels: Int

    /// LayerNormのepsilon
    public let layerNormEps: Float

    /// 位置埋め込みタイプ（"absolute" | "sincos"）
    public let positionEmbeddingType: String

    /// QKVバイアスを使用するか
    public let qkvBias: Bool

    /// MLP比率（hiddenSize * mlpRatio = intermediateSize）
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
}
```

**設計ノート**:
- `Sendable`準拠でSwift 6.2 concurrency対応
- `CodingKeys`で`snake_case` ↔ `camelCase`変換
- プリセット設定は不要（HuggingFaceから読み込むため）

### 3.2 AIMv2Model

メインのビジョンエンコーダーモデル。推論専用。

```swift
import MLX
import MLXNN

/// AIMv2 Vision Encoder（推論専用）
public class AIMv2Model: Module {
    @ModuleInfo public var patchEmbed: PatchEmbed
    @ModuleInfo public var blocks: [TransformerBlock]
    @ModuleInfo public var norm: LayerNorm

    public let config: AIMv2Configuration
    @ParameterInfo public var posEmbed: MLXArray?  // 学習可能位置埋め込み（absoluteの場合）

    public init(config: AIMv2Configuration) { ... }

    /// 順伝播（推論）
    /// - Parameter pixels: 画像テンソル [B, C, H, W]
    /// - Returns: 特徴テンソル [B, N+1, D]
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray { ... }

    /// CLS特徴を抽出（分類タスク用）
    public func extractCLSFeature(_ pixels: MLXArray) -> MLXArray { ... }

    /// 全パッチ特徴を抽出（検出タスク用）
    public func extractPatchFeatures(_ pixels: MLXArray) -> MLXArray { ... }
}
```

**主要メソッド**:

| メソッド | 入力 | 出力 | 用途 |
|---------|------|------|------|
| `callAsFunction` | `[B, C, H, W]` | `[B, N+1, D]` | 全特徴取得 |
| `extractCLSFeature` | `[B, C, H, W]` | `[B, D]` | 画像分類 |
| `extractPatchFeatures` | `[B, C, H, W]` | `[B, N, D]` | 物体検出等 |

**処理フロー**:
```
入力画像 [B, C, H, W]
    ↓
PatchEmbed → [B, N, D]
    ↓
CLSトークン追加 → [B, N+1, D]
    ↓
位置埋め込み追加
    ↓
TransformerBlock x12
    ↓
LayerNorm
    ↓
出力特徴 [B, N+1, D]
```

### 3.3 AIMv2ModelLoader

HuggingFaceからモデルをロードするローダー。

```swift
import Foundation
import MLX

/// HuggingFaceからAIMv2モデルをロード
public actor AIMv2ModelLoader {
    /// HuggingFace Hubからモデルをロード
    /// - Parameters:
    ///   - modelName: "apple/aimv2-large-patch14-224"形式
    ///   - useCache: ローカルキャッシュを使用
    /// - Returns: ロードされたモデルと設定
    public static func loadFromHub(
        modelName: String,
        useCache: Bool = true
    ) async throws -> (model: AIMv2Model, config: AIMv2Configuration)

    /// ローカルディレクトリからロード
    public static func loadFromDirectory(
        path: String
    ) throws -> (model: AIMv2Model, config: AIMv2Configuration)
}
```

**ロードフロー**:
```
1. HubAPIでモデルファイルをダウンロード
   ├── config.json
   └── model.safetensors (または複数の.safetensorsファイル)

2. config.jsonをデコード
   └── AIMv2Configuration作成

3. モデルインスタンス作成（ランダム重み）
   └── AIMv2Model(config: config)

4. safetensorsから重みをロード
   └── [String: MLXArray] 辞書

5. 重みをMLX形式に変換（sanitize）
   └── Conv2dのトランスポーズ等

6. モデルに重みを適用
   └── model.update(parameters:)

7. 評価モードに設定
   └── eval(model)

8. 完成したモデルを返す
```

### 3.4 ImagePreprocessor

画像の前処理を行うユーティリティ。

```swift
import CoreGraphics
import MLX

/// AIMv2用画像前処理
public struct AIMv2ImagePreprocessor: Sendable {
    public let imageSize: Int
    public let mean: [Float] = [0.485, 0.456, 0.406]  // ImageNet
    public let std: [Float] = [0.229, 0.224, 0.225]

    public init(imageSize: Int)

    /// 画像を前処理
    /// - Parameter image: 入力画像
    /// - Returns: [1, 3, H, W] 形式のテンソル
    public func preprocess(_ image: CGImage) -> MLXArray
}
```

**前処理ステップ**:
```
1. リサイズ（短辺をimageSizeにリサイズ）
2. センタークロップ（imageSize x imageSize）
3. RGB正規化: (pixel / 255.0 - mean) / std
4. テンソル変換: [H, W, 3] → [1, 3, H, W]
```

### 3.5 公開API

シンプルに使えるラッパーAPI。

```swift
import MLX

/// Swift-AIMの簡易API
public class SwiftAIM {
    private let model: AIMv2Model
    private let config: AIMv2Configuration
    private let preprocessor: AIMv2ImagePreprocessor

    /// モデルをロード
    public static func load(modelName: String) async throws -> SwiftAIM

    /// 画像から特徴を抽出
    public func extractFeatures(from image: CGImage) -> MLXArray

    /// CLS特徴を抽出
    public func extractCLSFeature(from image: CGImage) -> MLXArray
}
```

---

## 4. レイヤー実装詳細

### 4.1 PatchEmbed

画像を14x14ピクセルのパッチに分割し、埋め込みベクトルに変換。

```swift
import MLX
import MLXNN

public class PatchEmbed: Module, UnaryLayer {
    @ModuleInfo public var projection: Conv2d

    public let imageSize: Int
    public let patchSize: Int
    public let numPatches: Int

    public init(
        imageSize: Int = 224,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 768
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)

        self.projection = Conv2d(
            inChannels: inChannels,
            outChannels: embedDim,
            kernelSize: (patchSize, patchSize),
            stride: (patchSize, patchSize),
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // [B, C, H, W] -> [B, embedDim, H', W']
        let x = projection(x)

        let B = x.shape[0]
        let D = x.shape[1]

        // [B, D, H', W'] -> [B, H'*W', D]
        let x = x.reshaped(B, D, -1)  // [B, D, N]
        return x.transposed(0, 2, 1)  // [B, N, D]
    }
}
```

**計算例** (224x224画像):
```
入力: [1, 3, 224, 224]
  ↓ Conv2d(kernel=14, stride=14)
[1, 768, 16, 16]  (224/14 = 16パッチ)
  ↓ reshape + transpose
[1, 256, 768]  (16x16 = 256パッチ)
```

### 4.2 Attention

マルチヘッドセルフアテンション機構。

```swift
import MLX
import MLXNN
import MLXFast

public class Attention: Module {
    @ModuleInfo public var qkv: Linear
    @ModuleInfo public var proj: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dim: Int, numHeads: Int = 8, qkvBias: Bool = false) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self.qkv = Linear(dim, dim * 3, bias: qkvBias)
        self.proj = Linear(dim, dim)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]
        let N = x.shape[1]
        let D = numHeads * headDim

        // QKV projection
        let qkv = self.qkv(x)
            .reshaped(B, N, 3, numHeads, headDim)
            .transposed(2, 0, 3, 1, 4)

        let q = qkv[0]  // [B, numHeads, N, headDim]
        let k = qkv[1]
        let v = qkv[2]

        // Scaled dot-product attention
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        // Reshape and project
        let out = attn
            .transposed(0, 2, 1, 3)
            .reshaped(B, N, D)

        return proj(out)
    }
}
```

**アテンション計算**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

1. QKV射影: [B, N, D] → [B, N, 3D]
2. ヘッド分割: [B, N, 3D] → [B, H, N, D_h] x 3
3. スケールドドット積: softmax(QK^T / √D_h)
4. 重み付け和: Attention × V
5. ヘッド結合: [B, H, N, D_h] → [B, N, D]
6. 出力射影: [B, N, D] → [B, N, D]
```

### 4.3 MLP

フィードフォワードネットワーク（2層線形変換 + GELU活性化）。

```swift
import MLX
import MLXNN

public class MLP: Module {
    @ModuleInfo public var fc1: Linear
    @ModuleInfo public var fc2: Linear

    public init(dim: Int, hiddenDim: Int) {
        self.fc1 = Linear(dim, hiddenDim)
        self.fc2 = Linear(hiddenDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = fc1(x)
        x = MLXNN.gelu(x)
        x = fc2(x)
        return x
    }
}
```

**MLP構造**:
```
[B, N, D] → Linear(D, 4D) → GELU → Linear(4D, D) → [B, N, D]
```

### 4.4 TransformerBlock

Pre-normアーキテクチャのTransformerブロック。

```swift
import MLX
import MLXNN

public class TransformerBlock: Module {
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var attn: Attention
    @ModuleInfo public var norm2: LayerNorm
    @ModuleInfo public var mlp: MLP

    public init(
        dim: Int,
        numHeads: Int,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = false
    ) {
        self.norm1 = LayerNorm(dimensions: dim)
        self.attn = Attention(dim: dim, numHeads: numHeads, qkvBias: qkvBias)
        self.norm2 = LayerNorm(dimensions: dim)
        self.mlp = MLP(dim: dim, hiddenDim: Int(Float(dim) * mlpRatio))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm with residual
        var x = x
        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))
        return x
    }
}
```

**Pre-norm構造**:
```
x = x + Attention(LayerNorm(x))  # アテンション部
x = x + MLP(LayerNorm(x))        # MLP部
```

これはPost-norm（`LayerNorm(x + Attention(x))`）より学習が安定する。

---

## 5. モデルローディング

### 5.1 HubAPI

HuggingFace Hubからモデルファイルをダウンロード。

```swift
import Foundation

/// HuggingFace Hubからのモデルダウンロードを管理（スレッドセーフ）
public actor HubAPI {
    private var downloadCache: [String: URL] = [:]

    public init() {}

    /// モデルをダウンロード
    /// - Parameters:
    ///   - repo: リポジトリ名（"apple/aimv2-large-patch14-224"）
    ///   - useCache: キャッシュを使用
    /// - Returns: モデルディレクトリのURL
    public func downloadModel(
        repo: String,
        useCache: Bool = true
    ) async throws -> URL {
        // 1. メモリキャッシュチェック（actor isolationで保護）
        if useCache, let cached = downloadCache[repo] {
            return cached
        }

        // 2. ディスクキャッシュチェック
        let cacheDir = getCacheDirectory(for: repo)
        if useCache && fileExists(at: cacheDir) {
            downloadCache[repo] = cacheDir
            return cacheDir
        }

        // 3. HuggingFace APIでファイル一覧取得
        let files = try await fetchFileList(repo: repo)

        // 4. 必要ファイルをダウンロード
        for file in files {
            if file.hasSuffix(".safetensors") ||
               file == "config.json" ||
               file == "preprocessor_config.json" {
                try await downloadFile(repo: repo, filename: file, to: cacheDir)
            }
        }

        // 5. キャッシュに保存
        downloadCache[repo] = cacheDir
        return cacheDir
    }

    private func getCacheDirectory(for repo: String) -> URL {
        // ~/.cache/huggingface/hub/models--apple--aimv2-large-patch14-224/
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let repoPath = repo.replacingOccurrences(of: "/", with: "--")
        return homeDir
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--\(repoPath)")
    }

    private func fileExists(at url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    private func fetchFileList(repo: String) async throws -> [String] {
        // HuggingFace API実装（実際の実装は省略）
        fatalError("Not yet implemented")
    }

    private func downloadFile(repo: String, filename: String, to directory: URL) async throws {
        // ダウンロード実装（実際の実装は省略）
        fatalError("Not yet implemented")
    }
}
```

### 5.2 WeightLoader

safetensorsファイルから重みを読み込む。

```swift
import Foundation
import MLX

/// safetensorsファイルから重みをロードするユーティリティ
public struct WeightLoader {
    /// ディレクトリ内の全safetensorsファイルをロード
    /// - Parameter directory: モデルディレクトリのURL
    /// - Returns: テンソル名とMLXArrayの辞書
    public static func loadSafetensors(from directory: URL) throws -> [String: MLXArray] {
        var allWeights = [String: MLXArray]()

        // ディレクトリ内の.safetensorsファイルを探す
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        let safetensorsFiles = files.filter { $0.pathExtension == "safetensors" }

        // 各ファイルをロード（MLX組み込み関数を使用）
        for fileURL in safetensorsFiles {
            let data = try Data(contentsOf: fileURL)

            // ✅ MLX Swiftの組み込み関数でsafetensorsを直接ロード
            let weights = try MLX.loadArrays(data: data)

            // 重みをマージ（重複時は新しい方を優先）
            allWeights.merge(weights) { _, new in new }
        }

        return allWeights
    }
}
```

**実装ノート**:
- `MLX.loadArrays(data:)`はsafetensorsフォーマットを自動的にパースする
- 手動でヘッダーやメタデータを解析する必要はない
- 複数のsafetensorsファイルがある場合は全てマージされる

### 5.3 Weight Sanitization

PyTorch重みをMLX形式に変換。

```swift
extension AIMv2Model {
    /// PyTorch重みをMLX形式に変換
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // 1. プレフィックス除去
            newKey = newKey.replacingOccurrences(of: "vision_model.", with: "")
            newKey = newKey.replacingOccurrences(of: "encoder.", with: "")

            // 2. Conv2d重みのトランスポーズ
            // PyTorch: [out_channels, in_channels, kernel_h, kernel_w]
            // MLX:     [out_channels, kernel_h, kernel_w, in_channels]
            if newKey.contains("patch_embed.projection.weight") {
                newValue = value.transposed(0, 2, 3, 1)
            }

            // 3. キー名の変換（MLXの命名規則に合わせる）
            newKey = convertKeyName(newKey)

            sanitized[newKey] = newValue
        }

        return sanitized
    }

    private static func convertKeyName(_ key: String) -> String {
        // 例: "blocks.0.attn.qkv.weight" → "blocks.0.attn.qkv.weight"
        // 基本的にはそのまま使えるが、必要に応じて変換
        return key
    }
}
```

**重み変換の必要性**:
- PyTorchとMLXでテンソル次元順序が異なる場合がある
- Conv2dが最も顕著（最後の次元がMLXではin_channels）
- Linearは両方とも`[out_features, in_features]`なので変換不要

---

## 6. 画像前処理

### 6.1 ImagePreprocessor実装

```swift
import CoreGraphics
import CoreImage
import MLX

public struct AIMv2ImagePreprocessor: Sendable {
    public let imageSize: Int
    public let mean: [Float] = [0.485, 0.456, 0.406]  // ImageNet mean
    public let std: [Float] = [0.229, 0.224, 0.225]   // ImageNet std

    public init(imageSize: Int) {
        self.imageSize = imageSize
    }

    /// 画像を前処理してMLXArrayに変換
    public func preprocess(_ image: CGImage) -> MLXArray {
        // 1. リサイズ＆センタークロップ
        let cropped = resizeAndCenterCrop(image, targetSize: imageSize)

        // 2. ピクセルデータ取得 [H, W, 3]
        let pixels = extractPixels(from: cropped)

        // 3. 正規化: (x/255 - mean) / std
        let normalized = normalizePixels(pixels)

        // 4. テンソル変換: [H, W, 3] → [1, 3, H, W]
        let tensor = MLXArray(normalized)
            .transposed(2, 0, 1)           // [3, H, W]
            .expandedDimensions(axis: 0)   // [1, 3, H, W]

        return tensor
    }

    private func resizeAndCenterCrop(_ image: CGImage, targetSize: Int) -> CGImage {
        let width = image.width
        let height = image.height

        // 短辺をtargetSizeにリサイズ
        let scale: CGFloat
        if width < height {
            scale = CGFloat(targetSize) / CGFloat(width)
        } else {
            scale = CGFloat(targetSize) / CGFloat(height)
        }

        let newWidth = Int(CGFloat(width) * scale)
        let newHeight = Int(CGFloat(height) * scale)

        // リサイズ
        let context = CGContext(
            data: nil,
            width: newWidth,
            height: newHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        let resized = context.makeImage()!

        // センタークロップ
        let cropX = (newWidth - targetSize) / 2
        let cropY = (newHeight - targetSize) / 2
        let cropRect = CGRect(x: cropX, y: cropY, width: targetSize, height: targetSize)

        return resized.cropping(to: cropRect)!
    }

    private func extractPixels(from image: CGImage) -> [[[Float]]] {
        let width = image.width
        let height = image.height

        // ピクセルデータ取得
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!

        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else {
            fatalError("Failed to get pixel data")
        }

        let buffer = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        // [H, W, 3] 配列に変換
        var pixels = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0, count: 3), count: width),
            count: height
        )

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                pixels[y][x][0] = Float(buffer[offset]) / 255.0     // R
                pixels[y][x][1] = Float(buffer[offset + 1]) / 255.0 // G
                pixels[y][x][2] = Float(buffer[offset + 2]) / 255.0 // B
            }
        }

        return pixels
    }

    private func normalizePixels(_ pixels: [[[Float]]]) -> [[[Float]]] {
        let height = pixels.count
        let width = pixels[0].count

        var normalized = pixels

        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    normalized[y][x][c] = (pixels[y][x][c] - mean[c]) / std[c]
                }
            }
        }

        return normalized
    }
}
```

### 6.2 前処理パイプライン図

```
入力画像（任意サイズ）
    ↓
短辺リサイズ（targetSize）
    ↓
センタークロップ（targetSize x targetSize）
    ↓
RGB抽出 [H, W, 3]
    ↓
0-1正規化（÷255）
    ↓
ImageNet正規化
  R: (x - 0.485) / 0.229
  G: (x - 0.456) / 0.224
  B: (x - 0.406) / 0.225
    ↓
テンソル変換 [H, W, 3] → [1, 3, H, W]
    ↓
出力テンソル
```

---

## 7. 実装計画

### Phase 1: 基盤実装（1-2日）

**目標**: コアレイヤーとモデル構造の実装

- [ ] Package.swiftにMLX依存関係を追加
- [ ] `AIMv2Configuration.swift`を実装
- [ ] `PatchEmbed.swift`を実装
- [ ] `Attention.swift`を実装
- [ ] `MLP.swift`を実装
- [ ] `TransformerBlock.swift`を実装
- [ ] `AIMv2Model.swift`骨格を実装（forward passのみ）

**成果物**: 基本的なモデル構造が動作（ランダム重み）

### Phase 2: モデルローディング（2-3日）

**目標**: HuggingFaceからモデルをロードできるようにする

- [ ] `HubAPI.swift`を実装（ダウンロード機能）
- [ ] `WeightLoader.swift`を実装（safetensors読み込み）
- [ ] `WeightConverter.swift`を実装（sanitize機能）
- [ ] `AIMv2ModelLoader.swift`を実装
- [ ] ローディングの統合テスト

**成果物**: HuggingFaceモデルをロードして推論可能

### Phase 3: 前処理＆統合（1-2日）

**目標**: 使いやすいAPIの提供

- [ ] `ImagePreprocessor.swift`を実装
- [ ] `SwiftAIM.swift`（公開API）を実装
- [ ] サンプルコード作成
- [ ] 基本的なユニットテスト

**成果物**: エンドツーエンドで動作する推論パイプライン

### Phase 4: テスト＆ドキュメント（1-2日）

**目標**: 品質保証とドキュメント整備

- [ ] 各レイヤーのユニットテスト
- [ ] 統合テスト（PyTorchとの出力比較）
- [ ] パフォーマンステスト
- [ ] README.md作成
- [ ] API ドキュメント整備
- [ ] サンプルコード拡充

**成果物**: プロダクション準備完了

---

## 8. 使用例

### 8.1 シンプルAPI

```swift
import SwiftAIM
import CoreGraphics

// モデルをロード
let aim = try await SwiftAIM.load(modelName: "apple/aimv2-large-patch14-224")

// 画像をロード
guard let image = loadImage(from: "photo.jpg") else {
    fatalError("Failed to load image")
}

// 特徴抽出
let features = aim.extractFeatures(from: image)
print("Features shape:", features.shape)  // [1, 257, 768]

// CLS特徴のみ取得（画像分類等に使用）
let clsFeature = aim.extractCLSFeature(from: image)
print("CLS feature shape:", clsFeature.shape)  // [1, 768]
```

### 8.2 詳細API

```swift
import SwiftAIM
import MLX

// モデルとConfigを明示的にロード
let (model, config) = try await AIMv2ModelLoader.loadFromHub(
    modelName: "apple/aimv2-large-patch14-224"
)

print("Loaded model config:")
print("  - Hidden size:", config.hiddenSize)
print("  - Num layers:", config.numHiddenLayers)
print("  - Image size:", config.imageSize)

// 前処理を手動で行う
let preprocessor = AIMv2ImagePreprocessor(imageSize: config.imageSize)
let pixels = preprocessor.preprocess(image)

// 推論実行
let output = model(pixels)
print("Output shape:", output.shape)  // [1, 257, 768]

// CLS特徴を抽出
let clsToken = output[0..., 0, 0...]  // [1, 768]

// パッチ特徴を抽出（物体検出等に使用）
let patchFeatures = output[0..., 1..., 0...]  // [1, 256, 768]
```

### 8.3 バッチ処理

```swift
import SwiftAIM
import MLX

let aim = try await SwiftAIM.load(modelName: "apple/aimv2-large-patch14-224")

// 複数画像をバッチ処理
let images: [CGImage] = loadImages(from: ["img1.jpg", "img2.jpg", "img3.jpg"])

let preprocessor = AIMv2ImagePreprocessor(imageSize: 224)
let pixelBatch = images.map { preprocessor.preprocess($0) }
let batchTensor = MLX.concatenated(pixelBatch, axis: 0)  // [3, 3, 224, 224]

// バッチ推論
let batchFeatures = aim.model(batchTensor)
print("Batch features shape:", batchFeatures.shape)  // [3, 257, 768]
```

### 8.4 異なる解像度のモデル

```swift
// 336px解像度モデル（より詳細な特徴）
let aim336 = try await SwiftAIM.load(modelName: "apple/aimv2-large-patch14-336")
let features336 = aim336.extractFeatures(from: image)
print(features336.shape)  // [1, 577, 768]  (336/14)^2 + 1 = 577

// 448px解像度モデル（最も詳細）
let aim448 = try await SwiftAIM.load(modelName: "apple/aimv2-large-patch14-448")
let features448 = aim448.extractFeatures(from: image)
print(features448.shape)  // [1, 1025, 768]  (448/14)^2 + 1 = 1025
```

---

## 9. テスト戦略

このプロジェクトは**Swift Testing**フレームワーク（Swift 6.0+）を使用します。

### Swift Testing の特徴

- **モダンな構文**: `@Test`マクロで関数をテストとして定義
- **明示的なアサーション**: `#expect(...)`で期待値を検証
- **非同期サポート**: `async throws`が標準サポート
- **並列実行**: デフォルトでテストが並列実行される
- **タグ付け**: `@Test(.tags(.integration))`でテストを分類可能

### XCTestからの移行

| XCTest | Swift Testing |
|--------|---------------|
| `class MyTests: XCTestCase` | `// トップレベル関数` |
| `func testExample()` | `@Test func testExample()` |
| `XCTAssertEqual(a, b)` | `#expect(a == b)` |
| `XCTAssertTrue(condition)` | `#expect(condition)` |
| `XCTAssertNil(value)` | `#expect(value == nil)` |

### 9.1 ユニットテスト

各コンポーネントの動作を個別にテスト。

```swift
import Testing
@testable import SwiftAIM
import MLX

// PatchEmbedのテスト
@Test func testPatchEmbed() async throws {
    let patchEmbed = PatchEmbed(
        imageSize: 224,
        patchSize: 14,
        inChannels: 3,
        embedDim: 768
    )

    let input = MLXArray.zeros([1, 3, 224, 224])
    let output = patchEmbed(input)

    #expect(output.shape == [1, 256, 768])
}

// Attentionのテスト
@Test func testAttention() async throws {
    let attention = Attention(dim: 768, numHeads: 12)

    let input = MLXArray.zeros([1, 257, 768])
    let output = attention(input)

    #expect(output.shape == [1, 257, 768])
}

// Configurationのテスト
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
}
```

### 9.2 統合テスト

エンドツーエンドでの動作をテスト。

```swift
@Test func testModelLoading() async throws {
    // モデルをロード
    let (model, config) = try await AIMv2ModelLoader.loadFromHub(
        modelName: "apple/aimv2-large-patch14-224"
    )

    #expect(config.imageSize == 224)

    // ダミー画像で推論
    let dummyImage = MLXArray.zeros([1, 3, 224, 224])
    let output = model(dummyImage)

    #expect(output.shape == [1, 257, 768])
}

@Test func testImagePreprocessing() async throws {
    let preprocessor = AIMv2ImagePreprocessor(imageSize: 224)

    // テスト画像を作成
    let testImage = createTestImage(width: 300, height: 400)

    let tensor = preprocessor.preprocess(testImage)

    #expect(tensor.shape == [1, 3, 224, 224])
}
```

### 9.3 互換性テスト

PyTorchとの出力一致を確認。

```swift
@Test func testPyTorchCompatibility() async throws {
    // SwiftAIMで推論
    let (model, _) = try await AIMv2ModelLoader.loadFromHub(
        modelName: "apple/aimv2-large-patch14-224"
    )

    let input = loadFixedTestImage()  // 固定テスト画像
    let swiftOutput = model(input)

    // PyTorchの出力（事前計算済み）
    let expectedOutput = loadPyTorchOutput()

    // 許容誤差内で一致することを確認
    let diff = abs(swiftOutput - expectedOutput).max().item()
    #expect(diff < 1e-4, "Output differs from PyTorch by \(diff)")
}
```

### 9.4 パフォーマンステスト

推論速度とメモリ使用量を測定。

```swift
@Test func testInferencePerformance() async throws {
    let aim = try await SwiftAIM.load(modelName: "apple/aimv2-large-patch14-224")
    let testImage = createTestImage(width: 224, height: 224)

    // ウォームアップ
    _ = aim.extractFeatures(from: testImage)

    // 10回実行して平均時間を測定
    let startTime = Date()
    for _ in 0..<10 {
        _ = aim.extractFeatures(from: testImage)
    }
    let elapsed = Date().timeIntervalSince(startTime)
    let avgTime = elapsed / 10.0

    print("Average inference time: \(avgTime * 1000)ms")
    #expect(avgTime < 0.1, "Inference should be under 100ms")
}
```

---

## 10. レビュー指摘事項と修正

このセクションでは、設計レビューで指摘された問題点と、それに対する修正内容をまとめています。

### HIGH 優先度の修正

#### 1. TransformerBlock の `let x` 再宣言エラー

**問題**: Swiftでは同じスコープ内で`let`変数を再宣言できない。

**修正前**:
```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let x = x + attn(norm1(x))  // ❌ エラー
    let x = x + mlp(norm2(x))   // ❌ エラー
    return x
}
```

**修正後**:
```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x  // ✅ varで受ける
    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))
    return x
}
```

#### 2. TensorMetadata のタプル問題

**問題**: `Codable`はタプルを直接デコードできない。

**修正前**:
```swift
struct TensorMetadata: Codable {
    let dataOffsets: (Int, Int)  // ❌ デコード失敗
}
```

**修正後**: 配列で受け取り、バリデーションを追加
```swift
struct TensorMetadata: Codable {
    let dataOffsets: [Int]

    init(from decoder: Decoder) throws {
        // ... デコード処理
        guard dataOffsets.count == 2 else {
            throw DecodingError.dataCorruptedError(...)
        }
    }

    var offsetStart: Int { dataOffsets[0] }
    var offsetEnd: Int { dataOffsets[1] }
}
```

**さらに改善**: MLX.loadArrays()を使用することで、この構造体自体が不要に

#### 3. posEmbed の @ParameterInfo 欠落

**問題**: `@ParameterInfo`がないため、`model.update(parameters:)`で位置埋め込みが更新されない。

**修正前**:
```swift
public let posEmbed: MLXArray?  // ❌ パラメータとして認識されない
```

**修正後**:
```swift
@ParameterInfo public var posEmbed: MLXArray?  // ✅ パラメータとして登録
```

#### 4. WeightLoader の過剰な実装

**問題**: 手動でsafetensorsをパースする必要はない。MLX Swiftに組み込み関数がある。

**修正前**: 60行以上の手動パース実装

**修正後**:
```swift
public static func loadSafetensors(from directory: URL) throws -> [String: MLXArray] {
    var allWeights = [String: MLXArray]()
    let files = try FileManager.default.contentsOfDirectory(...)

    for fileURL in files.filter({ $0.pathExtension == "safetensors" }) {
        let data = try Data(contentsOf: fileURL)
        let weights = try MLX.loadArrays(data: data)  // ✅ 組み込み関数
        allWeights.merge(weights) { _, new in new }
    }

    return allWeights
}
```

### MEDIUM 優先度の修正

#### 5. テストフィルターコマンドの誤り

**修正前**:
```bash
swift test --filter swift_aimTests.example  # ❌ ターゲット名が違う
```

**修正後**:
```bash
swift test --filter SwiftAIMTests.testModelLoading  # ✅ 正しい構文
```

### LOW 優先度の修正

#### 6. HubAPI の actor 宣言の無意味さ

**問題**: `actor`宣言なのに全メソッドが`static`では、actor isolationの恩恵がない。

**修正前**:
```swift
public actor HubAPI {
    public static func downloadModel(...) { }  // ❌ actorの意味がない
}
```

**修正後**:
```swift
public actor HubAPI {
    private var downloadCache: [String: URL] = [:]  // ✅ 状態を保持

    public func downloadModel(...) async throws -> URL {
        // actor isolationでキャッシュを保護
        if let cached = downloadCache[repo] { return cached }
        ...
    }
}
```

### その他の改善

#### Swift Testing の採用

- XCTestではなく、Swift Testing（Swift 6.0+）を使用
- `@Test`マクロと`#expect()`でモダンなテストコード
- 並列実行とタグ付けをサポート

---

## 付録

### A. AIMv2モデル一覧

| モデル名 | パラメータ数 | 解像度 | パッチ数 | 用途 |
|---------|------------|--------|----------|------|
| aimv2-large-patch14-224 | 0.3B | 224 | 256 | 汎用 |
| aimv2-large-patch14-336 | 0.3B | 336 | 576 | 高精度 |
| aimv2-large-patch14-448 | 0.3B | 448 | 1024 | 最高精度 |
| aimv2-huge-patch14-224 | 0.7B | 224 | 256 | 大規模 |
| aimv2-1B-patch14-224 | 1B | 224 | 256 | 超大規模 |
| aimv2-3B-patch14-224 | 3B | 224 | 256 | 最大規模 |

### B. 設定パラメータ

| パラメータ | Large | Huge | 1B | 3B |
|-----------|-------|------|----|----|
| hidden_size | 768 | 1024 | 1536 | 2048 |
| num_hidden_layers | 12 | 24 | 24 | 32 |
| num_attention_heads | 12 | 16 | 24 | 32 |
| intermediate_size | 3072 | 4096 | 6144 | 8192 |

### C. 参考リンク

- **AIMv2リポジトリ**: https://github.com/apple/ml-aim
- **MLX Swift**: https://github.com/ml-explore/mlx-swift
- **MLX Swift Examples**: https://github.com/ml-explore/mlx-swift-examples
- **HuggingFace Collection**: https://huggingface.co/collections/apple/aimv2
- **DeepWiki**: https://deepwiki.com/apple/ml-aim

---

**文書バージョン**: 2.0
**最終更新**: 2025-10-28
**作成者**: Claude Code
