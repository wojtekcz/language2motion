import TensorFlow
import Datasets
import TextModels
import TranslationModels


// Transformer with LangMotionBatch

public struct LangMotionTransformerConfig: ModelConfig {
    public let vocabSize: Int
    public let nbJoints: Int
    public let nbMixtures: Int
    public let layerCount: Int
    public let encoderDepth: Int
    public let decoderDepth: Int
    public let feedForwardSize: Int
    public let headCount: Int
    public let dropoutProbability: Double
    public let sentenceMaxPositionalLength: Int
    public let motionMaxPositionalLength: Int
    public let motionPositionalEncodingSize: Int // TODO: kill motionPositionalEncodingSize
    public let encoderSelfAttentionTemp: Double
    public let decoderSourceAttentionTemp: Double
    public let decoderSelfAttentionTemp: Double

    public init(vocabSize: Int, nbJoints: Int, nbMixtures: Int, layerCount: Int, encoderDepth: Int, decoderDepth: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double,
                sentenceMaxPositionalLength: Int, motionMaxPositionalLength: Int, motionPositionalEncodingSize: Int,
                encoderSelfAttentionTemp: Double, decoderSourceAttentionTemp: Double, decoderSelfAttentionTemp: Double) {
        self.vocabSize = vocabSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.layerCount = layerCount
        self.encoderDepth = encoderDepth
        self.decoderDepth = decoderDepth
        self.feedForwardSize = feedForwardSize
        self.headCount = headCount
        self.dropoutProbability = dropoutProbability
        self.sentenceMaxPositionalLength = sentenceMaxPositionalLength
        self.motionMaxPositionalLength = motionMaxPositionalLength
        self.motionPositionalEncodingSize = motionPositionalEncodingSize
        self.encoderSelfAttentionTemp = encoderSelfAttentionTemp
        self.decoderSourceAttentionTemp = decoderSourceAttentionTemp
        self.decoderSelfAttentionTemp = decoderSelfAttentionTemp
    }
}

public struct LangMotionTransformerOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var preds: MixtureModelPreds
    public var encoded: EncoderOutput<Scalar>
    public var decoded: DecoderOutput<Scalar>

    @differentiable
    public init(preds: MixtureModelPreds, encoded: EncoderOutput<Scalar>, decoded: DecoderOutput<Scalar>) {
        self.preds = preds
        self.encoded = encoded
        self.decoded = decoded
    }
}


public struct LangMotionTransformer: Module {

    @noDerivative public var config: LangMotionTransformerConfig

    // encoding sentence
    public var embedding: Embedding<Float>
    public var positionalEncoding: PositionalEncoding
    public var encoder: Encoder

    // decoding motion
    public var motionDense: Dense<Float>
    public var motionPositionalEncoding: PositionalEncoding
    public var motionNorm: LayerNorm<Float>
    public var decoder: Decoder

    // generating motion
    public var mixtureModel: MotionGaussianMixtureModel

    public init(config: LangMotionTransformerConfig) {
        self.config = config

        // encoding sentence
        self.embedding = Embedding<Float>(vocabularySize: config.vocabSize, embeddingSize: config.encoderDepth, embeddingsInitializer: glorotUniform())
        self.positionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)

        let encAttention = MultiHeadAttention(sourceSize: config.encoderDepth, targetSize: config.encoderDepth,
                                              headCount: config.headCount, headSize: config.encoderDepth/config.headCount,
                                              attentionDropoutProbability: Float(config.dropoutProbability), matrixResult: false)
        
        let encFeedForward = PositionwiseFeedForward(dimensionalityModel: config.encoderDepth,
                                                     innerLayerDimensionality: config.feedForwardSize)

        self.encoder = Encoder(layer: .init(size: config.encoderDepth, selfAttention: encAttention, feedForward: encFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)

        // decoding motion
        self.motionDense = Dense<Float>(inputSize: config.nbJoints, outputSize: config.decoderDepth)
        self.motionPositionalEncoding = PositionalEncoding(size: config.decoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)
        self.motionNorm = LayerNorm(featureCount: config.decoderDepth, axis: 2)

        let decSelfAttention = MultiHeadAttention(sourceSize: config.decoderDepth, targetSize: config.decoderDepth,
                                                  headCount: config.headCount, headSize: config.decoderDepth/config.headCount,
                                                  attentionDropoutProbability: Float(config.dropoutProbability), matrixResult: false)
        
        let decSourceAttention = MultiHeadAttention(sourceSize: config.decoderDepth, targetSize: config.encoderDepth,
                                                    headCount: config.headCount, headSize: config.decoderDepth/config.headCount,
                                                    attentionDropoutProbability: Float(config.dropoutProbability), matrixResult: false)
        
        let decFeedForward = PositionwiseFeedForward(dimensionalityModel: config.decoderDepth,
                                                     innerLayerDimensionality: config.feedForwardSize)

        self.decoder = Decoder(layer: .init(size: config.decoderDepth, selfAttention: decSelfAttention, sourceAttention: decSourceAttention, feedForward: decFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount, derivativeAllLayers: true)

        // generating motion
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: config.decoderDepth*config.layerCount, nbJoints: config.nbJoints, nbMixtures: config.nbMixtures)
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> LangMotionTransformerOutput<Float> {
        let encoded = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        // reformat decoded.allOutputs[] into one tensor
        let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
        let rslt = LangMotionTransformerOutput(preds: self.mixtureModel(mixtureModelInput), encoded: encoded, decoded: decoded)
        return rslt
    }
    
    @differentiable
    public func encode(input: LangMotionBatch.Sentence) -> EncoderOutput<Float> {
        let embedded = self.positionalEncoding(self.embedding(input.tokenIds))
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask, selfAttentionTemperature: Float(config.encoderSelfAttentionTemp))
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(sourceMask: Tensor<Float>, motionPart: LangMotionBatch.MotionPart, memory: Tensor<Float>) -> DecoderOutput<Float> {
        // start flag, pos enc, current motion, padding with motion
        let shape = motionPart.motion.shape
        let (origBatchSize, numFrames, nbJoints) = (shape[0], shape[1], shape[2])

        // squeeze all frames in a batch into first dimension
        let tmpMotionFrames = motionPart.motion.reshaped(to: [origBatchSize*numFrames, nbJoints])
        let tmpMotionFeatures = motionDense(tmpMotionFrames) // 47 -> decoderDepth
        // unsqueeze back to bs x numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, numFrames, config.decoderDepth])

        motionFeatures = motionPositionalEncoding(motionFeatures)
        // TODO: add segment encoding
        motionFeatures = motionNorm(motionFeatures)

        let decoderInput = DecoderInput(sequence: motionFeatures, sourceMask: sourceMask, targetMask: motionPart.mask, memory: memory,
                                        sourceAttentionTemperature: Float(config.decoderSourceAttentionTemp), selfAttentionTemperature: Float(config.decoderSelfAttentionTemp))
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(config: LangMotionTransformerConfig, encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>,
                positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding,
                mixtureModel: MotionGaussianMixtureModel, motionDense: Dense<Float>, motionNorm: LayerNorm<Float>) {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.mixtureModel = mixtureModel
        self.motionDense = motionDense
        self.motionNorm = motionNorm
    }
}
