// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MotionDataset",
    platforms: [
        .macOS(.v10_13),
    ],
    products: [
        .library(name: "Checkpoints", targets: ["Checkpoints"]),
        .library(name: "Batcher", targets: ["Batcher"]),
        .library(name: "Datasets", targets: ["Datasets"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "TextModels", targets: ["TextModels"]),
        .library(name: "MotionLangModels", targets: ["MotionLangModels"]),
        .library(name: "LangMotionModels", targets: ["LangMotionModels"]),
        .library(name: "SummaryWriter", targets: ["SummaryWriter"]),
        .library(name: "TranslationModels", targets: ["TranslationModels"]),
        .library(name: "TrainingLoop", targets: ["TrainingLoop"]),
        .executable(name: "PreprocessMotionDataset", targets: ["PreprocessMotionDataset"]),
        .executable(name: "Motion2lang", targets: ["Motion2lang"]),
        .executable(name: "Lang2motion", targets: ["Lang2motion"]),
        .executable(name: "Lang2motionSet", targets: ["Lang2motionSet"]),
    ],
    dependencies: [
        .package(name: "SwiftProtobuf", url: "https://github.com/apple/swift-protobuf.git", from: "1.10.2")
    ],
    targets: [
        .testTarget(name: "MotionDatasetTests", dependencies: ["Datasets", "ModelSupport", "LangMotionModels", "SummaryWriter"]),
        .target(
            name: "Checkpoints", dependencies: ["SwiftProtobuf", "ModelSupport"],
            path: "Sources/Checkpoints"),
        .target(name: "PreprocessMotionDataset", dependencies: ["Datasets"], path: "Sources/PreprocessMotionDataset"),
        .target(name: "Batcher", path: "Sources/Batcher"),
        .target(name: "Datasets", dependencies: ["ModelSupport", "Batcher"], path: "Sources/Datasets"),
        .target(
            name: "ModelSupport", dependencies: ["SwiftProtobuf", "STBImage"], path: "Sources/Support",
            exclude: ["STBImage"]),
        .target(name: "STBImage", path: "Sources/Support/STBImage"),
        .target(name: "TextModels", dependencies: ["Checkpoints", "Datasets"], path: "Sources/Models/Text"),
        .target(name: "MotionLangModels", dependencies: ["Datasets", "TextModels", "ModelSupport", "TranslationModels", "TrainingLoop"], path: "Sources/Models/MotionLang"),
        .target(name: "LangMotionModels", dependencies: ["Datasets", "TextModels", "ModelSupport", "TranslationModels", "TrainingLoop"], path: "Sources/Models/LangMotion"),
        .target(name: "SummaryWriter", path: "Sources/SummaryWriter"),
        .target(name: "TranslationModels", dependencies: ["TextModels", "Datasets"], path: "Sources/Models/Translation"),
        .target(name: "TrainingLoop", dependencies: ["ModelSupport"], path: "Sources/TrainingLoop"),
        .target(
            name: "Motion2lang",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter", "MotionLangModels", "TrainingLoop"],
            path: "Sources/Motion2lang"),
        .target(
            name: "Lang2motion",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter", "LangMotionModels", "TrainingLoop"],
            path: "Sources/Lang2motion"),
        .target(
            name: "Lang2motionSet",
            dependencies: ["TranslationModels", "TextModels", "Datasets", "ModelSupport", "SummaryWriter", "LangMotionModels", "TrainingLoop"],
            path: "Sources/Lang2motionSet"),
    ]
)
