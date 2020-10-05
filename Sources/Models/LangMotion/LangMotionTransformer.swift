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
    public let motionPositionalEncodingSize: Int
    public let encoderSelfAttentionTemp: Double
    public let decoderSourceAttentionTemp: Double
    public let decoderSelfAttentionTemp: Double

    public init(vocabSize: Int, nbJoints: Int, nbMixtures: Int, layerCount: Int, encoderDepth: Int, decoderDepth: Int,
                feedForwardSize: Int, headCount: Int, dropoutProbability: Double, sentenceMaxPositionalLength: Int, motionMaxPositionalLength: Int, motionPositionalEncodingSize: Int,
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

    public var encoder: Encoder
    public var decoder: Decoder
    public var embedding: Embedding<Float>
    public var positionalEncoding: PositionalEncoding
    public var motionPositionalEncoding: PositionalEncoding
    public var motionNorm: LayerNorm<Float>
    public var mixtureModel: MotionGaussianMixtureModel
    @noDerivative public var config: LangMotionTransformerConfig

    public init(config: LangMotionTransformerConfig) {

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

        self.positionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
        self.motionPositionalEncoding = PositionalEncoding(size: config.motionPositionalEncodingSize, dropoutProbability: config.dropoutProbability, maxLength: config.motionMaxPositionalLength)
        self.embedding = Embedding<Float>(vocabularySize: config.vocabSize, embeddingSize: config.encoderDepth, embeddingsInitializer: glorotUniform())
        
        self.encoder = Encoder(layer: .init(size: config.encoderDepth, selfAttention: encAttention, feedForward: encFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)
        self.decoder = Decoder(layer: .init(size: config.decoderDepth, selfAttention: decSelfAttention, sourceAttention: decSourceAttention, feedForward: decFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount, derivativeAllLayers: true)
        self.motionNorm = LayerNorm(featureCount: config.decoderDepth, axis: 2)
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: config.decoderDepth*config.layerCount, nbJoints: config.nbJoints, nbMixtures: config.nbMixtures)
        self.config = config
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
        var motionPartFeatures: Tensor<Float>

        // start flag, pos enc, current motion, padding with motion
        let shape = motionPart.motion.shape
        let (batchSize, numFrames) = (shape[0], shape[1])

        // motion positional encoding
        var motionPositionalEncodingVector = Tensor<Float>(repeating: 0.0, shape: [batchSize, numFrames, config.motionPositionalEncodingSize])
        motionPositionalEncodingVector = motionPositionalEncoding(motionPositionalEncodingVector)
        
        // compute padding
        let paddingSize = config.decoderDepth - (1 + config.motionPositionalEncodingSize + config.nbJoints)
        
        let multiplyBy = paddingSize/config.nbJoints + 1
        let motionFramePadding = motionPart.motion.tiled(multiples: [1, 1, multiplyBy])[0..., 0..., 0..<paddingSize]

        // stack everything together
        let tensorStack = [motionPart.startFlag, motionPositionalEncodingVector, motionPart.motion, motionFramePadding]
        let tmpMotionPartFeatures = Tensor<Float>(concatenating: tensorStack, alongAxis: 2)
        motionPartFeatures = tmpMotionPartFeatures

        motionPartFeatures = self.motionNorm(motionPartFeatures)

        let decoderInput = DecoderInput(sequence: motionPartFeatures, sourceMask: sourceMask, targetMask: motionPart.mask, memory: memory, 
                                        sourceAttentionTemperature: Float(config.decoderSourceAttentionTemp), selfAttentionTemperature: Float(config.decoderSelfAttentionTemp))
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(config: LangMotionTransformerConfig, encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>,
                positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding,
                mixtureModel: MotionGaussianMixtureModel, motionNorm: LayerNorm<Float>) {
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.mixtureModel = mixtureModel
        self.motionNorm = motionNorm
    }
}
