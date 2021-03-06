import Foundation
import TensorFlow
import Checkpoints
import TranslationModels
import TextModels
import ModelSupport


extension MotionCatDistHead {
    public init(reader: CheckpointReader, config: LangMotionCatDistTransformerConfig, scope: String) {
        self.init(
            inputSize: config.decoderDepth,
            nbJoints: config.nbJoints,
            discreteBins: config.discreteBins,
            catDistWeights: Dense<Float>(reader: reader, config: config, scope: scope + "/catDistWeights"),
            linearStop: Dense<Float>(reader: reader, config: config, scope: scope + "/linearStop"),
            norm: LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: 2, epsilon: 0.001)
        )
    }
}


extension LangMotionCatDistTransformer {
    public init(checkpoint: URL, config: LangMotionCatDistTransformerConfig, name: String) throws {
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
            let _langEmbedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/langEmbedding")
            let _langPositionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
            let _encoder = Encoder(reader: reader, config: config, scope: scope + "/encoder", activation: config.activation.actFunc())

            // decoding
            let _jointEmbedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/jointEmbedding")
            let _motionDense = Dense<Float>(reader: reader, config: config, scope: scope + "/motionDense", activation: config.activation.actFunc())
            let _motionPositionalEncoding = PositionalEncoding(size: config.decoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)
            let _motionSegmentEmbedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/motionSegmentEmbedding")
            let _motionNorm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/motionNorm", axis: 2, epsilon: 0.001)
            let _decoder = Decoder(reader: reader, config: config, derivativeAllLayers: true, scope: scope + "/decoder", activation: config.activation.actFunc())
            
            // generating
            let _catDistHead = MotionCatDistHead(reader: reader, config: config, scope: scope + "/catDistHead")

            self.init(config: config,
                      langEmbedding: _langEmbedding, langPositionalEncoding: _langPositionalEncoding, encoder: _encoder,
                      jointEmbedding: _jointEmbedding, motionDense: _motionDense, motionPositionalEncoding: _motionPositionalEncoding,
                      motionSegmentEmbedding: _motionSegmentEmbedding, motionNorm: _motionNorm, decoder: _decoder,
                      catDistHead: _catDistHead)
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load LangMotionTransformer from checkpoint. \(error)")
            throw error
        }
    }
}
