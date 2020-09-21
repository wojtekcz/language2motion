import Foundation
import TensorFlow
import Checkpoints
import TranslationModels
import TextModels
import ModelSupport

extension LangMotionTransformer: ExportableLayer {
    public var nameMappings: [String: String] {        
        [
            "encoder": "encoder",
            "decoder": "decoder",
            "embedding": "embedding",
            "mixtureModel": "mixtureModel",
            "motionNorm": "motionNorm"
        ]
    }
}

extension MotionGaussianMixtureModel: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "linearMixtureMeans": "linearMixtureMeans",
            "linearMixtureVars": "linearMixtureVars",
            "linearMixtureWeights": "linearMixtureWeights",
            "linearStop": "linearStop",
            // inputSize: Int
            // nbJoints: Int
            // nbMixtures: Int
            // outputSize: Int
        ] 
    }
}

extension LangMotionTransformer {
    public func writeCheckpoint(to location: URL, name: String) throws {
        print("saving model \(location.path)")
        try TranslationModels.writeCheckpoint(self, to: location, name: name)
    }
}
