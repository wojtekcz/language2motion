// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MotionDataset",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "ImageClassificationModels", targets: ["ImageClassificationModels"]),
        .library(name: "Batcher", targets: ["Batcher"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .library(name: "MotionModels", targets: ["MotionModels"]),
        .library(name: "SummaryWriter", targets: ["SummaryWriter"]),
        .executable(name: "PreprocessMotionData", targets: ["PreprocessMotionData"]),
        .executable(name: "ResNet-img2label", targets: ["ResNet-img2label"]),
        .executable(name: "ResNet-motion2label", targets: ["ResNet-motion2label"]),
        .executable(name: "BERT-language2label", targets: ["BERT-language2label"]),
        .executable(name: "Transformer-motion2label2", targets: ["Transformer-motion2label2"]),
        .executable(name: "Transformer2-motion2label2", targets: ["Transformer2-motion2label2"])
    ],
    dependencies: [
        .package(name: "SwiftProtobuf", url: "https://github.com/apple/swift-protobuf.git", from: "1.9.0")
    ],
    targets: [
        .testTarget(name: "MotionDatasetTests", dependencies: ["Datasets"]),
        .target(name: "PreprocessMotionData", dependencies: ["Datasets"], path: "Sources/PreprocessMotionData"),
        .target(
            name: "ResNet-img2label", dependencies: ["ImageClassificationModels", "Datasets"],
            path: "Sources/ResNet-img2label"),
        .target(
            name: "ResNet-motion2label", dependencies: ["ImageClassificationModels", "Datasets", "SummaryWriter"],
            path: "Sources/ResNet-motion2label"),
        .target(
            name: "BERT-language2label", dependencies: ["TextModels", "Datasets", "SummaryWriter"],
            path: "Sources/BERT-language2label"),
        .target(
            name: "Transformer-motion2label2", dependencies: ["ImageClassificationModels", "TextModels", "Datasets", "ModelSupport", "MotionModels", "SummaryWriter"],
            path: "Sources/Transformer-motion2label2"),
        .target(
            name: "Transformer2-motion2label2", dependencies: ["ImageClassificationModels", "TextModels", "Datasets", "ModelSupport", "MotionModels", "SummaryWriter"],
            path: "Sources/Transformer2-motion2label2"),
        .target(name: "Batcher", path: "Sources/Batcher"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Sources/Datasets"),
        .target(name: "ImageClassificationModels", path: "Sources/Models/ImageClassification"),
        .target(
            name: "ModelSupport", dependencies: ["SwiftProtobuf", "STBImage"], path: "Sources/Support",
            exclude: ["STBImage"]),
        .target(name: "STBImage", path: "Sources/Support/STBImage"),
        .target(name: "TextModels", dependencies: ["Datasets"], path: "Sources/Models/Text"),
        .target(name: "MotionModels", dependencies: ["Datasets", "TextModels", "ModelSupport", "ImageClassificationModels"], path: "Sources/Models/Motion"),
        .target(name: "SummaryWriter", path: "Sources/SummaryWriter")
    ]
)
