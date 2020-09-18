import Foundation
import TensorFlow
import Checkpoints
import TranslationModels
import TextModels
import ModelSupport


protocol InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            weight: reader.readTensor(name: scope + "/weight"),
            bias: reader.readTensor(name: scope + "/bias"),
            activation: identity
        )
    }
}

extension Embedding: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            embeddings: reader.readTensor(name: scope + "/embeddings")
        )
    }
}

extension MotionGaussianMixtureModel: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            inputSize: config.modelSize,
            nbJoints: config.nbJoints,
            nbMixtures: config.nbMixtures,
            linearMixtureMeans: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureMeans"),
            linearMixtureVars: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureVars"),
            linearMixtureWeights: Dense<Float>(reader: reader, config: config, scope: scope + "/linearMixtureWeights"),
            linearStop: Dense<Float>(reader: reader, config: config, scope: scope + "/linearStop")
        )
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            offset: reader.readTensor(name: scope + "/offset"),
            scale: reader.readTensor(name: scope + "/scale"),
            axis: 2,
            epsilon: 0.001)
        // FIXME: serialize/deserialize axis & epsilon defaults
        print("Shouldn't call this LayerNorm initializer, axis & epsilon are not serializable yet.")
    }
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String, axis: Int, epsilon: Scalar) {
        self.init(
            offset: reader.readTensor(name: scope + "/offset"),
            scale: reader.readTensor(name: scope + "/scale"),
            axis: axis,
            epsilon: epsilon)
    }
}

extension MultiHeadAttention: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            sourceSize: config.modelSize,
            targetSize: config.modelSize,
            headCount: config.headCount,
            headSize: config.modelSize/config.headCount,
            queryActivation: identity,
            keyActivation: identity,
            valueActivation: identity,
            attentionDropoutProbability: 0,
            matrixResult: false,
            queryWeight: reader.readTensor(name: scope + "/queryWeight"),
            queryBias: reader.readTensor(name: scope + "/queryBias"),
            keyWeight: reader.readTensor(name: scope + "/keyWeight"),
            keyBias: reader.readTensor(name: scope + "/keyBias"),
            valueWeight: reader.readTensor(name: scope + "/valueWeight"),
            valueBias: reader.readTensor(name: scope + "/valueBias")
        )
    }
}

extension PositionwiseFeedForward: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            dense1: Dense<Float>(reader: reader, config: config, scope: scope + "/dense1"),
            dense2: Dense<Float>(reader: reader, config: config, scope: scope + "/dense2"),
            dropout: Dropout<Float>(probability: config.dropoutProbability)
        )
    }
}

extension SublayerConnection: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        self.init(
            norm: LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: -1, epsilon: 1e-6),
            dropout: Dropout<Float>(probability: config.dropoutProbability)
        )
    }
}

extension TransformerEncoderLayer2: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        let _selfAttention = MultiHeadAttention(reader: reader, config: config, scope: scope + "/selfAttention")
        let _feedForward = PositionwiseFeedForward(reader: reader, config: config, scope: scope + "/feedForward")
        let _sublayers = (0..<2).map { i in
            SublayerConnection(reader: reader, config: config, scope: scope + "/sublayers/SublayerConnection_h\(i)")
        }
        self.init(
            selfAttention: _selfAttention, 
            feedForward: _feedForward, 
            sublayers: _sublayers
        )
    }
}

extension Encoder: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        let _layers = (0..<config.layerCount).map { i in
            TransformerEncoderLayer2(reader: reader, config: config, scope: scope + "/layers/TransformerEncoderLayer2_h\(i)")
        }
        let _norm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: 2, epsilon: 0.001)
        self.init(layers: _layers, norm: _norm)
    }
}



extension TransformerDecoderLayer: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        let _selfAttention = MultiHeadAttention(reader: reader, config: config, scope: scope + "/selfAttention")
        let _sourceAttention = MultiHeadAttention(reader: reader, config: config, scope: scope + "/sourceAttention")
        let _feedForward = PositionwiseFeedForward(reader: reader, config: config, scope: scope + "/feedForward")
        let _sublayers = (0..<3).map { i in
            SublayerConnection(reader: reader, config: config, scope: scope + "/sublayers/SublayerConnection_h\(i)")
        }
        self.init(
            selfAttention: _selfAttention, 
            sourceAttention: _sourceAttention,
            feedForward: _feedForward, 
            sublayers: _sublayers
        )
    }
}

extension Decoder: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: LangMotionTransformerConfig, scope: String) {
        let _layers = (0..<config.layerCount).map { i in
            TransformerDecoderLayer(reader: reader, config: config, scope: scope + "/layers/TransformerDecoderLayer_h\(i)")
        }
        let _norm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: 2, epsilon: 0.001)
        self.init(layers: _layers, norm: _norm)
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
            let _encoder = Encoder(reader: reader, config: config, scope: scope + "/encoder")
            let _decoder = Decoder(reader: reader, config: config, scope: scope + "/decoder")
            let _embedding = Embedding<Float>(reader: reader, config: config, scope: scope + "/embedding")
            let _positionalEncoding = PositionalEncoding(size: config.modelSize, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
            let motionPositionalEncodingSize = 32
            let _motionPositionalEncoding = PositionalEncoding(size: motionPositionalEncodingSize, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)
            let _sourceEmbed = Sequential(_embedding, _positionalEncoding)

            let _mixtureModel = MotionGaussianMixtureModel(reader: reader, config: config, scope: scope + "/mixtureModel")

            let _motionNorm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/motionNorm", axis: 2, epsilon: 0.001)
            
            self.init(config: config, encoder: _encoder, decoder: _decoder, embedding: _embedding, positionalEncoding: _positionalEncoding,
                      motionPositionalEncoding: _motionPositionalEncoding, sourceEmbed: _sourceEmbed, mixtureModel: _mixtureModel, motionNorm: _motionNorm)
        } catch {
            // If checkpoint is invalid, throw the error and exit.
            print("Fail to load LangMotionTransformer from checkpoint. \(error)")
            throw error
        }
    }
}
