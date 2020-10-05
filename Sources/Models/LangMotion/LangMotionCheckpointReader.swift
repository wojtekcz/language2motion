import Foundation
import TensorFlow
import Checkpoints
import TranslationModels
import TextModels
import ModelSupport


extension MotionGaussianMixtureModel {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            inputSize: config.decoderDepth,
            nbJoints: config.nbJoints,
            nbMixtures: config.nbMixtures,
            linearMixtureMeans: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureMeans"),
            linearMixtureVars: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureVars"),
            linearMixtureWeights: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureWeights"),
            linearStop: Dense<Float>(reader: reader, config: config, scope: scope + "/linearStop")
        )
    }
}

extension LangMotionTransformer {
    public init(checkpoint: URL, config: LangMotionTransformerConfig, name: String) throws {
        print("Loading model \"\(name)\" from \"\(checkpoint.path)\"...")
        // Try loading from the given checkpoint.
        do {
            // create reader
            let auxiliary: [String] = [
                "hparams.json"
            ]

            let reader: CheckpointReader = try CheckpointReader(
                checkpointLocation: checkpoint.appendingPathComponent(name),
                modelName: name,
                additionalFiles: auxiliary)
            
            // TODO: load config (values)
            
            // load objects            
            let scope = "model"

            // encoding
            let _embedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/embedding")
            let _positionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
            let _encoder = Encoder(reader: reader, config: config, scope: scope + "/encoder")

            // decoding
            let _motionDense = Dense<Float>(reader: reader, config: config, scope: scope + "/motionDense")
            
            let motionPositionalEncodingSize = 32 // FIXME: update motionPositionalEncoding
            let _motionPositionalEncoding = PositionalEncoding(size: motionPositionalEncodingSize, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)

            let _motionSegmentEmbedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/segmentEmbedding")
            let _motionNorm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/motionNorm", axis: 2, epsilon: 0.001)
            let _decoder = Decoder(reader: reader, config: config, derivativeAllLayers: true, scope: scope + "/decoder")
            
            let _mixtureModel = MotionGaussianMixtureModel(reader: reader, config: config, scope: scope + "/mixtureModel")

            self.init(config: config,
                      embedding: _embedding, positionalEncoding: _positionalEncoding, encoder: _encoder,
                      motionDense: _motionDense, motionPositionalEncoding: _motionPositionalEncoding,
                      motionSegmentEmbedding: _motionSegmentEmbedding, motionNorm: _motionNorm, decoder: _decoder,
                      mixtureModel: _mixtureModel)
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load LangMotionTransformer from checkpoint. \(error)")
            throw error
        }
    }
}