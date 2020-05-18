// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MotionDataset",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "MotionDataset", targets: ["MotionDataset"]),
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "Batcher", targets: ["Batcher"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .executable(name: "RunPreprocess", targets: ["RunPreprocess"]),
        .executable(name: "ResNet-img2label", targets: ["ResNet-img2label"]),
        .executable(name: "BERT-language2label", targets: ["BERT-language2label"])
    ],
    dependencies: [
        .package(name: "SwiftProtobuf", url: "https://github.com/apple/swift-protobuf.git", from: "1.7.0"),
    ],
    targets: [
        .target(name: "MotionDataset", path: "Sources/MotionDataset"),
        .testTarget(name: "MotionDatasetTests", dependencies: ["MotionDataset"]),
        .target(name: "RunPreprocess", dependencies: ["MotionDataset"], path: "Sources/RunPreprocess"),
        .target(
            name: "ResNet-img2label", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Sources/ResNet-img2label"),
        .target(
            name: "BERT-language2label", dependencies: ["TextModels", "Datasets"],
            path: "Sources/BERT-language2label"),
        .target(name: "Batcher", path: "Sources/Batcher"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Sources/Datasets"),
        .target(name: "ImageClassificationModels", path: "Sources/Models/ImageClassification"),
        .target(
            name: "ModelSupport", dependencies: ["SwiftProtobuf", "STBImage"], path: "Sources/Support",
            exclude: ["STBImage"]),
        .target(name: "STBImage", path: "Sources/Support/STBImage"),
        .target(name: "TextModels", dependencies: ["Datasets"], path: "Sources/Models/Text"),
    ]
)
