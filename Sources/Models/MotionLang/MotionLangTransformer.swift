import TensorFlow
import Datasets
import TranslationModels
import TextModels


// Transformer with MotionLangBatch

public struct MotionLangTransformerConfig {
    public let vocabSize: Int
    public let nbJoints: Int
    public let layerCount: Int
    public let modelSize: Int
    public let feedForwardSize: Int
    public let headCount: Int
    public let dropoutProbability: Double

    public init(vocabSize: Int, nbJoints: Int, layerCount: Int, modelSize: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double) {
        self.vocabSize = vocabSize
        self.nbJoints = nbJoints
        self.layerCount = layerCount
        self.modelSize = modelSize
        self.feedForwardSize = feedForwardSize
        self.headCount = headCount
        self.dropoutProbability = dropoutProbability
    }
}

public struct MotionLangTransformer: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var positionalEncoding: PositionalEncoding
    public var motionNorm: LayerNorm<Float>
    public var motionDense: Dense<Float>
    public var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    public var generator: Generator
    @noDerivative public var config: MotionLangTransformerConfig
    
    public init(config: MotionLangTransformerConfig) {
        let attention = MultiHeadAttention(sourceSize: config.modelSize,
                                           targetSize: config.modelSize,
                                           headCount: config.headCount,
                                           headSize: config.modelSize/config.headCount,
                                           matrixResult: false)
        let feedForward = PositionwiseFeedForward(dimensionalityModel: config.modelSize,
                                                  innerLayerDimensionality: config.feedForwardSize)
        
        positionalEncoding = PositionalEncoding(size: config.modelSize,
                                                dropoutProbability: config.dropoutProbability)
        
        self.encoder = Encoder(layer: .init(size: config.modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)
        self.decoder = Decoder(layer: .init(size: config.modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)
        self.motionNorm = LayerNorm(featureCount: config.modelSize, axis: 2)
        self.motionDense = Dense<Float>(inputSize: config.nbJoints, outputSize: config.modelSize)
        self.targetEmbed = Sequential(Embedding(vocabularySize: config.vocabSize, embeddingSize: config.modelSize, embeddingsInitializer: glorotUniform()), positionalEncoding)
        self.generator = Generator(dimModel: config.modelSize, vocabSize: config.vocabSize)
        self.config = config
    }
    
    @differentiable
    public func callAsFunction(_ input: MotionLangBatch.MLSource) -> Tensor<Float> {
        let encoded = self.encode(input: input)
        let decoded = self.decode(input: input, memory: encoded.lastLayerOutput)
        let rslt = self.generator(decoded.lastLayerOutput)
        return rslt
    }
    
    @differentiable
    public func encode(input: MotionLangBatch.MLSource) -> EncoderOutput<Float> {
        let origBatchSize = input.motion.shape[0]
        let length = input.motion.shape[1]
        let numFrames = input.motion.shape[2]
        let hiddenSize = self.config.modelSize

        let tmpBatchSize = origBatchSize * length
        let tmpMotionFrames = input.motion.reshaped(to: [tmpBatchSize, numFrames])

        let tmpMotionFeatures = motionDense(tmpMotionFrames) // batch size here is origBatchSize*numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, length, hiddenSize])
        motionFeatures = self.motionNorm(motionFeatures)

        motionFeatures = positionalEncoding(motionFeatures)

        let encoderInput: TransformerInput = TransformerInput(sequence: motionFeatures, attentionMask: input.mask, selfAttentionTemperature: 1.0)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(input: MotionLangBatch.MLSource, memory: Tensor<Float>) -> DecoderOutput<Float> {
        let embedded = self.targetEmbed(input.targetTokenIds)
        let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.mask, targetMask: input.targetMask, memory: memory, sourceAttentionTemperature: 1.0, selfAttentionTemperature: 1.0)
        return self.decoder(decoderInput)
    }
    
    @differentiable
    public func generate(input: Tensor<Float>) -> Tensor<Float> {
        self.generator(input)
    }
}
