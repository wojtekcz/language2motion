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
    public let mixtureDepth: Int

    public init(vocabSize: Int, nbJoints: Int, nbMixtures: Int, layerCount: Int, encoderDepth: Int, decoderDepth: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double,
                sentenceMaxPositionalLength: Int, motionMaxPositionalLength: Int, mixtureDepth: Int) {
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
        self.mixtureDepth = mixtureDepth
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
    public var langEmbedding: Embedding<Float>
    public var langPositionalEncoding: PositionalEncoding
    public var encoder: Encoder

    // decoding motion
    public var motionDense: Dense<Float>
    public var motionPositionalEncoding: PositionalEncoding
    public var motionSegmentEmbedding: Embedding<Float>
    public var motionNorm: LayerNorm<Float>
    public var decoder: Decoder

    // generating motion
    public var preMixtureDense: Dense<Float>
    public var mixtureModel: MotionGaussianMixtureModel

    public init(config: LangMotionTransformerConfig) {
        self.config = config

        // encoding sentence
        self.langEmbedding = Embedding<Float>(vocabularySize: config.vocabSize, embeddingSize: config.encoderDepth, embeddingsInitializer: glorotUniform())
        self.langPositionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
        
        let encAttention = MultiHeadAttention(sourceSize: config.encoderDepth, targetSize: config.encoderDepth,
                                              headCount: config.headCount, headSize: config.encoderDepth/config.headCount,
                                              attentionDropoutProbability: Float(config.dropoutProbability), matrixResult: false)
        
        let encFeedForward = PositionwiseFeedForward(dimensionalityModel: config.encoderDepth,
                                                     innerLayerDimensionality: config.feedForwardSize)

        self.encoder = Encoder(layer: .init(size: config.encoderDepth, selfAttention: encAttention, feedForward: encFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)

        // decoding motion
        self.motionDense = Dense<Float>(inputSize: config.nbJoints, outputSize: config.decoderDepth, activation: swish)
        self.motionPositionalEncoding = PositionalEncoding(size: config.decoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)

        // The token type vocabulary will always be small and so we use the one-hot approach here
        // as it is always faster for small vocabularies.
        let initializerStandardDeviation: Float = 0.02
        self.motionSegmentEmbedding = Embedding<Float>(
            vocabularySize: LangMotionBatch.MotionSegment.allCases.count,
            embeddingSize: config.decoderDepth,
            embeddingsInitializer: truncatedNormalInitializer(standardDeviation: Tensor<Float>(initializerStandardDeviation)))

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
        self.preMixtureDense = Dense<Float>(inputSize: config.decoderDepth, outputSize: config.mixtureDepth, activation: swish)
        //config.decoderDepth*config.layerCount
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: config.mixtureDepth, nbJoints: config.nbJoints, nbMixtures: config.nbMixtures)
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> LangMotionTransformerOutput<Float> {
        let encoded = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        // reformat decoded.allOutputs[] into one tensor
        //let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
        let mixtureModelInput = self.preMixtureDense(decoded.lastLayerOutput)
        let rslt = LangMotionTransformerOutput(preds: self.mixtureModel(mixtureModelInput), encoded: encoded, decoded: decoded)
        return rslt
    }
    
    @differentiable
    public func encode(input: LangMotionBatch.Sentence, encoderSelfAttentionTemp: Float = 1.0) -> EncoderOutput<Float> {
        let embedded = self.langPositionalEncoding(self.langEmbedding(input.tokenIds))
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.selfAttentionMask, selfAttentionTemperature: encoderSelfAttentionTemp)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(sourceMask: Tensor<Float>, motionPart: LangMotionBatch.MotionPart, memory: Tensor<Float>,
                       decoderSourceAttentionTemp: Float = 1.0, decoderSelfAttentionTemp: Float = 1.0) -> DecoderOutput<Float> {
        // start flag, pos enc, current motion, padding with motion
        let shape = motionPart.motion.shape
        let (origBatchSize, numFrames, nbJoints) = (shape[0], shape[1], shape[2])

        // squeeze all frames in a batch into first dimension
        let tmpMotionFrames = motionPart.motion.reshaped(to: [origBatchSize*numFrames, nbJoints])
        let tmpMotionFeatures = motionDense(tmpMotionFrames) // 47 -> decoderDepth
        // unsqueeze back to bs x numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, numFrames, config.decoderDepth])

        motionFeatures = motionPositionalEncoding(motionFeatures)
        let segmentEmbeddings = motionSegmentEmbedding(motionPart.segmentIDs[0..., 0..., 0])
        motionFeatures = motionFeatures + segmentEmbeddings
        motionFeatures = motionNorm(motionFeatures)
        
        let decoderInput = DecoderInput(sequence: motionFeatures, sourceMask: sourceMask, targetMask: motionPart.decSelfAttentionMask, memory: memory,
                                        sourceAttentionTemperature: decoderSourceAttentionTemp, selfAttentionTemperature: decoderSelfAttentionTemp)
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(config: LangMotionTransformerConfig,
                langEmbedding: Embedding<Float>, langPositionalEncoding: PositionalEncoding, encoder: Encoder,
                motionDense: Dense<Float>, motionPositionalEncoding: PositionalEncoding, motionSegmentEmbedding: Embedding<Float>,
                motionNorm: LayerNorm<Float>, decoder: Decoder, preMixtureDense: Dense<Float>,
                mixtureModel: MotionGaussianMixtureModel
    ) {
        self.config = config

        self.langEmbedding = langEmbedding
        self.langPositionalEncoding = langPositionalEncoding
        self.encoder = encoder

        self.motionDense = motionDense
        self.motionPositionalEncoding = motionPositionalEncoding
        self.motionSegmentEmbedding = motionSegmentEmbedding
        self.motionNorm = motionNorm
        self.decoder = decoder

        self.mixtureModel = mixtureModel
        self.preMixtureDense = preMixtureDense
    }
}
