import Foundation
import TensorFlow
import Checkpoints
import TranslationModels

extension MotionLangTransformer {

    public init(checkpoint: URL, config: MotionLangTransformerConfig, name: String) throws {
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
            let _encoder = Encoder(reader: reader, config: config, scope: scope + "/encoder")
            let _decoder = Decoder(reader: reader, config: config, scope: scope + "/decoder")
            let _embedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/embedding")
            let _positionalEncoding = PositionalEncoding(size: config.modelSize, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)

            let _motionPositionalEncoding = PositionalEncoding(size: config.modelSize, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)

            let _motionNorm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/motionNorm", axis: 2, epsilon: 0.001)
            let _motionDense = Dense<Float>(reader: reader, config: config, scope: scope + "/motionDense")
            
            let _generator = Generator(reader: reader, config: config, scope: scope + "/generator")
            
            self.init(config: config, encoder: _encoder, decoder: _decoder, embedding: _embedding, positionalEncoding: _positionalEncoding, motionPositionalEncoding: _motionPositionalEncoding, motionNorm: _motionNorm, motionDense: _motionDense, generator: _generator)
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load LangMotionTransformer from checkpoint. \(error)")
            throw error
        }
    }
}
