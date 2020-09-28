import TensorFlow
import Datasets
import TranslationModels
import TextModels


// Transformer with MotionLangBatch

public struct MotionLangTransformerConfig: ModelConfig {
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

    public init(vocabSize: Int, nbJoints: Int, layerCount: Int, encoderDepth: Int, decoderDepth: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double,
                sentenceMaxPositionalLength: Int, motionMaxPositionalLength: Int) {
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
    }
}

public struct MotionLangTransformer: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var embedding: Embedding<Float>
    public var positionalEncoding: PositionalEncoding
    public var motionPositionalEncoding: PositionalEncoding
    public var motionNorm: LayerNorm<Float>
    public var motionDense: Dense<Float>
    public var generator: Generator
    @noDerivative public var config: MotionLangTransformerConfig
    
    public init(config: MotionLangTransformerConfig) {
        let encAttention = MultiHeadAttention(sourceSize: config.encoderDepth,
                                           targetSize: config.encoderDepth,
                                           headCount: config.headCount,
                                           headSize: config.encoderDepth/config.headCount,
                                           attentionDropoutProbability: Float(config.dropoutProbability),
                                           matrixResult: false)

        let decSelfAttention = MultiHeadAttention(sourceSize: config.decoderDepth,
                                           targetSize: config.decoderDepth,
                                           headCount: config.headCount,
                                           headSize: config.decoderDepth/config.headCount,
                                           attentionDropoutProbability: Float(config.dropoutProbability),
                                           matrixResult: false)

        let decSourceAttention = MultiHeadAttention(sourceSize: config.decoderDepth,
                                           targetSize: config.encoderDepth,
                                           headCount: config.headCount,
                                           headSize: config.decoderDepth/config.headCount,
                                           attentionDropoutProbability: Float(config.dropoutProbability),
                                           matrixResult: false)

        let encFeedForward = PositionwiseFeedForward(dimensionalityModel: config.encoderDepth,
                                                     innerLayerDimensionality: config.feedForwardSize)

        let decFeedForward = PositionwiseFeedForward(dimensionalityModel: config.decoderDepth,
                                                  innerLayerDimensionality: config.feedForwardSize)

        self.embedding = Embedding(vocabularySize: config.vocabSize, embeddingSize: config.decoderDepth, embeddingsInitializer: glorotUniform())
        self.positionalEncoding = PositionalEncoding(size: config.decoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
        self.motionPositionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)
        
        self.encoder = Encoder(layer: .init(size: config.encoderDepth, selfAttention: encAttention, feedForward: encFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)
        self.decoder = Decoder(layer: .init(size: config.decoderDepth, selfAttention: decSelfAttention, sourceAttention: decSourceAttention, feedForward: decFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)
        self.motionNorm = LayerNorm(featureCount: config.encoderDepth, axis: 2)
        self.motionDense = Dense<Float>(inputSize: config.nbJoints, outputSize: config.encoderDepth)
        self.generator = Generator(dimModel: config.decoderDepth, vocabSize: config.vocabSize)
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
        let encoderDepth = self.config.encoderDepth

        let tmpBatchSize = origBatchSize * length
        let tmpMotionFrames = input.motion.reshaped(to: [tmpBatchSize, numFrames])

        let tmpMotionFeatures = motionDense(tmpMotionFrames) // batch size here is origBatchSize*numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, length, encoderDepth])
        motionFeatures = self.motionNorm(motionFeatures)

        motionFeatures = motionPositionalEncoding(motionFeatures)

        let encoderInput: TransformerInput = TransformerInput(sequence: motionFeatures, attentionMask: input.mask, selfAttentionTemperature: 1.0)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(input: MotionLangBatch.MLSource, memory: Tensor<Float>) -> DecoderOutput<Float> {
        let embedded = self.positionalEncoding(self.embedding(input.targetTokenIds))
        let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.mask, targetMask: input.targetMask, memory: memory, sourceAttentionTemperature: 1.0, selfAttentionTemperature: 1.0)
        return self.decoder(decoderInput)
    }
    
    @differentiable
    public func generate(input: Tensor<Float>) -> Tensor<Float> {
        self.generator(input)
    }
}

extension MotionLangTransformer {

    public init(config: MotionLangTransformerConfig, encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>,
                positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding,
                motionNorm: LayerNorm<Float>, motionDense: Dense<Float>, generator: Generator) {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.motionNorm = motionNorm
        self.motionDense = motionDense
        self.generator = generator
    }
}
