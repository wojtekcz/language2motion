import Foundation
import TensorFlow
import Checkpoints
import TextModels

public protocol ModelConfig {
    var encoderDepth: Int { get }
    var decoderDepth: Int { get }
    var headCount: Int { get }
    var layerCount: Int { get }
    var dropoutProbability: Double { get }
}

public struct AttentionConfig {
    public let sourceSize: Int
    public let targetSize: Int
    public let headCount: Int
    public let headSize: Int
    public let dropoutProbability: Double
    
    public init (sourceSize: Int, targetSize: Int, headCount: Int, headSize: Int, dropoutProbability: Double) {
        self.sourceSize = sourceSize
        self.targetSize = targetSize
        self.headCount = headCount
        self.headSize = headSize
        self.dropoutProbability = dropoutProbability
    }
}

public protocol InitializableFromPythonCheckpoint {
    init(reader: CheckpointReader, config: ModelConfig, scope: String)
}

extension Dense: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            weight: reader.readTensor(name: scope + "/weight"),
            bias: reader.readTensor(name: scope + "/bias"),
            activation: identity
        )
    }
}

extension Embedding: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            embeddings: reader.readTensor(name: scope + "/embeddings")
        )
    }
}

extension LayerNorm: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            offset: reader.readTensor(name: scope + "/offset"),
            scale: reader.readTensor(name: scope + "/scale"),
            axis: 2,
            epsilon: 0.001)
        // FIXME: serialize/deserialize axis & epsilon defaults
        print("Shouldn't call this LayerNorm initializer, axis & epsilon are not serializable yet.")
    }
    public init(reader: CheckpointReader, config: ModelConfig, scope: String, axis: Int, epsilon: Scalar) {
        self.init(
            offset: reader.readTensor(name: scope + "/offset"),
            scale: reader.readTensor(name: scope + "/scale"),
            axis: axis,
            epsilon: epsilon)
    }
}

extension MultiHeadAttention {
    public init(reader: CheckpointReader, config: AttentionConfig, scope: String) {
        self.init(
            sourceSize: config.sourceSize,
            targetSize: config.targetSize,
            headCount: config.headCount,
            headSize: config.sourceSize/config.headCount,
            queryActivation: identity,
            keyActivation: identity,
            valueActivation: identity,
            attentionDropoutProbability: Float(config.dropoutProbability),
//            attentionDropoutProbability: 0,
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
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            dense1: Dense<Float>(reader: reader, config: config, scope: scope + "/dense1"),
            dense2: Dense<Float>(reader: reader, config: config, scope: scope + "/dense2"),
            dropout: Dropout<Float>(probability: config.dropoutProbability)
        )
    }
}

extension Generator: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            dense: Dense<Float>(reader: reader, config: config, scope: scope + "/dense")
        )
    }
}

extension SublayerConnection: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        self.init(
            norm: LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: -1, epsilon: 1e-6),
            dropout: Dropout<Float>(probability: config.dropoutProbability)
        )
    }
}

extension TransformerEncoderLayer2: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {

        let selfAttConfig = AttentionConfig(
            sourceSize: config.encoderDepth,
            targetSize: config.encoderDepth,
            headCount: config.headCount,
            headSize: config.encoderDepth/config.headCount,
            dropoutProbability: config.dropoutProbability
//            dropoutProbability: 0
        )
        
        let _selfAttention = MultiHeadAttention(reader: reader, config: selfAttConfig, scope: scope + "/selfAttention")
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
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        let _layers = (0..<config.layerCount).map { i in
            TransformerEncoderLayer2(reader: reader, config: config, scope: scope + "/layers/TransformerEncoderLayer2_h\(i)")
        }
        let _norm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: 2, epsilon: 0.001)
        self.init(layers: _layers, norm: _norm)
    }
}

extension TransformerDecoderLayer: InitializableFromPythonCheckpoint {
    public init(reader: CheckpointReader, config: ModelConfig, scope: String) {
        let selfAttConfig = AttentionConfig(
            sourceSize: config.decoderDepth,
            targetSize: config.decoderDepth,
            headCount: config.headCount,
            headSize: config.decoderDepth/config.headCount,
//            dropoutProbability: 0
            dropoutProbability: config.dropoutProbability
        )

        let sourceAttConfig = AttentionConfig(
            sourceSize: config.decoderDepth,
            targetSize: config.encoderDepth,
            headCount: config.headCount,
            headSize: config.decoderDepth/config.headCount,
//            dropoutProbability: 0
            dropoutProbability: config.dropoutProbability
        )

        let _selfAttention = MultiHeadAttention(reader: reader, config: selfAttConfig, scope: scope + "/selfAttention")
        let _sourceAttention = MultiHeadAttention(reader: reader, config: sourceAttConfig, scope: scope + "/sourceAttention")
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

extension Decoder {
    public init(reader: CheckpointReader, config: ModelConfig, derivativeAllLayers: Bool, scope: String) {
        let _layers = (0..<config.layerCount).map { i in
            TransformerDecoderLayer(reader: reader, config: config, scope: scope + "/layers/TransformerDecoderLayer_h\(i)")
        }
        let _norm = LayerNorm<Float>(reader: reader, config: config, scope: scope + "/norm", axis: 2, epsilon: 0.001)
        self.init(layers: _layers, norm: _norm, derivativeAllLayers: derivativeAllLayers)
    }
}