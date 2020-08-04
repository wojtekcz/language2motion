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
        .library(name: "MotionClassifiers", targets: ["MotionClassifiers"]),
        .library(name: "MotionLangModels", targets: ["MotionLangModels"]),
        .library(name: "LangMotionModels", targets: ["LangMotionModels"]),
        .library(name: "SummaryWriter", targets: ["SummaryWriter"]),
        .library(name: "TranslationModels", targets: ["TranslationModels"]),
        .executable(name: "PreprocessMotionDataset", targets: ["PreprocessMotionDataset"]),
        .executable(name: "Language2label", targets: ["Language2label"]),
        .executable(name: "Motion2label", targets: ["Motion2label"]),
        .executable(name: "Transformer-Translation", targets: ["Transformer-Translation"]),
        .executable(name: "Lang2lang", targets: ["Lang2lang"]),
        .executable(name: "Motion2lang", targets: ["Motion2lang"]),
        .executable(name: "Lang2motion", targets: ["Lang2motion"]),
    ],
    dependencies: [
        .package(name: "SwiftProtobuf", url: "https://github.com/apple/swift-protobuf.git", from: "1.10.2")
    ],
    targets: [
        .testTarget(name: "MotionDatasetTests", dependencies: ["Datasets"]),
        .target(name: "PreprocessMotionDataset", dependencies: ["Datasets"], path: "Sources/PreprocessMotionDataset"),
        .target(
            name: "Language2label", dependencies: ["TextModels", "Datasets", "SummaryWriter"],
            path: "Sources/Language2label"),
        .target(
            name: "Motion2label", dependencies: ["ImageClassificationModels", "TextModels", "Datasets", "ModelSupport", "MotionClassifiers", "SummaryWriter"],
            path: "Sources/Motion2label"),
        .target(name: "Batcher", path: "Sources/Batcher"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Sources/Datasets"),
        .target(name: "ImageClassificationModels", path: "Sources/Models/ImageClassification"),
        .target(
            name: "ModelSupport", dependencies: ["SwiftProtobuf", "STBImage"], path: "Sources/Support",
            exclude: ["STBImage"]),
        .target(name: "STBImage", path: "Sources/Support/STBImage"),
        .target(name: "TextModels", dependencies: ["Datasets"], path: "Sources/Models/Text"),
        .target(name: "MotionLangModels", dependencies: ["Datasets", "TextModels", "ModelSupport", "ImageClassificationModels", "TranslationModels"], path: "Sources/Models/MotionLang"),
        .target(name: "MotionClassifiers", dependencies: ["Datasets", "TextModels", "ModelSupport", "ImageClassificationModels", "TranslationModels"], path: "Sources/Models/MotionClassifiers"),
        .target(name: "LangMotionModels", dependencies: ["Datasets", "TextModels", "ModelSupport", "ImageClassificationModels", "TranslationModels"], path: "Sources/Models/LangMotion"),
        .target(name: "SummaryWriter", path: "Sources/SummaryWriter"),
        .target(name: "TranslationModels", dependencies: ["Datasets"], path: "Sources/Models/Translation"),
        .target(
            name: "Transformer-Translation",
            dependencies: ["TranslationModels", "Datasets", "ModelSupport"],
            path: "Sources/Transformer-Translation"),
        .target(
            name: "Lang2lang",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter"],
            path: "Sources/Lang2lang"),
        .target(
            name: "Motion2lang",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter", "MotionLangModels"],
            path: "Sources/Motion2lang"),
        .target(
            name: "Lang2motion",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter", "LangMotionModels"],
            path: "Sources/Lang2motion"),
    ]
)
