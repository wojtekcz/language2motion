import Foundation
import TensorFlow
import TextModels
import TranslationModels


public enum ActivationFunction: String, Codable {
    case swish
    case relu
    case identity
    
    public func actFunc() -> Activation<Float> {
        switch self {
        case .swish:
            return TensorFlow.swish
        case .relu:
            return TensorFlow.relu
        case .identity:
            return TensorFlow.identity
        }
    }
}


public struct LangMotionCatDistTransformerConfig: ModelConfig, Codable {
    public let vocabSize: Int
    public let nbJoints: Int
    public let layerCount: Int
    public let encoderDepth: Int
    public let decoderDepth: Int
    public let feedForwardSize: Int
    public let headCount: Int
    public let dropoutProbability: Double
    public let sentenceMaxPositionalLength: Int
    public let motionMaxPositionalLength: Int
    public let discreteBins: Int
    public let activation: ActivationFunction

    public init(vocabSize: Int, nbJoints: Int, layerCount: Int, encoderDepth: Int, decoderDepth: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double,
                sentenceMaxPositionalLength: Int, motionMaxPositionalLength: Int, discreteBins: Int,
                activation: ActivationFunction
    ) {
        self.vocabSize = vocabSize
        self.nbJoints = nbJoints
        self.layerCount = layerCount
        self.encoderDepth = encoderDepth
        self.decoderDepth = decoderDepth
        self.feedForwardSize = feedForwardSize
        self.headCount = headCount
        self.dropoutProbability = dropoutProbability
        self.sentenceMaxPositionalLength = sentenceMaxPositionalLength
        self.motionMaxPositionalLength = motionMaxPositionalLength
        self.discreteBins = discreteBins
        self.activation = activation
    }
    
    public static func createFromJSONURL(_ url: URL) -> Self? {
        let json = try! String(contentsOf: url, encoding: .utf8).data(using: .utf8)!
        guard let f = try? JSONDecoder().decode(LangMotionCatDistTransformerConfig.self, from: json) else {
            return nil
        }
        return f
    }
    
    public func write(to configURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        if let encodedData = try? encoder.encode(self) {
            try encodedData.write(to: configURL)
        }
    }
}
