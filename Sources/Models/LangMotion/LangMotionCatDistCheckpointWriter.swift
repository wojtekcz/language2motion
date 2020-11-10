import Foundation
import TensorFlow
import Checkpoints
import TranslationModels
import TextModels
import ModelSupport

extension LangMotionCatDistTransformer: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "langEmbedding": "langEmbedding",
            "encoder": "encoder",
            "jointEmbedding": "jointEmbedding",
            "motionDense": "motionDense",
            "motionSegmentEmbedding": "motionSegmentEmbedding",
            "motionNorm": "motionNorm",
            "decoder": "decoder",
            "catDistHead": "catDistHead"
        ]
    }
}

extension MotionCatDistHead: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "catDistWeights": "catDistWeights",
            "linearStop": "linearStop",
            "norm": "norm"
        ]
    }
}

extension LangMotionCatDistTransformer {
    public func writeCheckpoint(to location: URL, name: String) throws {
        print("saving model \(name) in \(location.path)")
        try TranslationModels.writeCheckpoint(self, to: location, name: name)
    }
}
