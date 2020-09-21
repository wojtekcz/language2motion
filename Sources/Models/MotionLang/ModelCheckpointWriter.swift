import Foundation
import TensorFlow
import Checkpoints
import TranslationModels

extension MotionLangTransformer: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "encoder": "encoder",
            "decoder": "decoder",
            "embedding": "embedding",
            "motionNorm": "motionNorm",
            "motionDense": "motionDense",
            "generator": "generator"
        ]
    }
}

extension MotionLangTransformer {
    public func writeCheckpoint(to location: URL, name: String) throws {
        print("saving model \(location.path)")
        try TranslationModels.writeCheckpoint(self, to: location, name: name)
    }
}
