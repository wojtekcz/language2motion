import TensorFlow
import Datasets
import TextModels
import TranslationModels


public struct LangMotionCatDistTransformerOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var preds: MotionCatDistPreds
    public var encoded: EncoderOutput<Scalar>
    public var decoded: DecoderOutput<Scalar>

    @differentiable
    public init(preds: MotionCatDistPreds, encoded: EncoderOutput<Scalar>, decoded: DecoderOutput<Scalar>) {
        self.preds = preds
        self.encoded = encoded
        self.decoded = decoded
    }
}


public struct LangMotionCatDistTransformer: Module {

    @noDerivative public var config: LangMotionCatDistTransformerConfig

    // encoding sentence
    public var langEmbedding: Embedding<Float>
    public var langPositionalEncoding: PositionalEncoding
    public var encoder: Encoder

    // decoding motion
    public var jointEmbedding: Embedding<Float>
    public var motionDense: Dense<Float>
    public var motionPositionalEncoding: PositionalEncoding
    public var motionSegmentEmbedding: Embedding<Float>
    public var motionNorm: LayerNorm<Float>
    public var decoder: Decoder

    // generating motion
    public var catDistHead: MotionCatDistHead

    public init(config: LangMotionCatDistTransformerConfig) {
        self.config = config

        // encoding sentence
        self.langEmbedding = Embedding<Float>(vocabularySize: config.vocabSize, embeddingSize: config.encoderDepth, embeddingsInitializer: glorotUniform())
        self.langPositionalEncoding = PositionalEncoding(size: config.encoderDepth, dropoutProbability: config.dropoutProbability, maxLength: config.sentenceMaxPositionalLength)
        
        let encAttention = MultiHeadAttention(sourceSize: config.encoderDepth, targetSize: config.encoderDepth,
                                              headCount: config.headCount, headSize: config.encoderDepth/config.headCount,
                                              attentionDropoutProbability: Float(config.dropoutProbability), matrixResult: false)
        
        let encFeedForward = PositionwiseFeedForward(dimensionalityModel: config.encoderDepth,
                                                     innerLayerDimensionality: config.feedForwardSize, activation: config.activation.actFunc())

        self.encoder = Encoder(layer: .init(size: config.encoderDepth, selfAttention: encAttention, feedForward: encFeedForward, dropoutProb: config.dropoutProbability), layerCount: config.layerCount)

        // decoding motion
        // TODO: parametrize jointEmbeddingSize
        let jointEmbeddingSize = 5
        self.jointEmbedding = Embedding<Float>(vocabularySize: config.discreteBins, embeddingSize: jointEmbeddingSize, embeddingsInitializer: glorotUniform())
        self.motionDense = Dense<Float>(inputSize: config.nbJoints*jointEmbeddingSize, outputSize: config.decoderDepth, activation: config.activation.actFunc())
//        self.motionDense = Dense<Float>(inputSize: config.nbJoints, outputSize: config.decoderDepth, activation: config.activation)
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
                                                     innerLayerDimensionality: config.feedForwardSize, activation: config.activation.actFunc())
        
        // TODO: parametrize kernel_size
        let kernel_size = 1
        let decConv1D = Conv1D<Float>(filterShape: (kernel_size, config.decoderDepth, config.decoderDepth), stride: 1, padding: .same, activation: config.activation.actFunc(), useBias: true)

        self.decoder = Decoder(layer: .init(size: config.decoderDepth, selfAttention: decSelfAttention, sourceAttention: decSourceAttention, feedForward: decFeedForward, dropoutProb: config.dropoutProbability, conv1D: decConv1D
        ), layerCount: config.layerCount, derivativeAllLayers: true)

        // generating motion
        self.catDistHead = MotionCatDistHead(inputSize: config.decoderDepth, nbJoints: config.nbJoints, discreteBins: config.discreteBins)
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> LangMotionCatDistTransformerOutput<Float> {
        let encoded = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        // reformat decoded.allOutputs[] into one tensor
        //let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
        let catDistHeadInput = decoded.lastLayerOutput
        let rslt = LangMotionCatDistTransformerOutput(preds: self.catDistHead(catDistHeadInput), encoded: encoded, decoded: decoded)
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
        let shape = motionPart.discreteMotion.shape
        let (origBatchSize, numFrames, nbJoints) = (shape[0], shape[1], shape[2])

        // squeeze all joint values in a batch into first dimension
        let jointEmbeddingSize = 5
        let embeddedJointValues = jointEmbedding(motionPart.discreteMotion.flattened().expandingShape(at: 0)).reshaped(to: [origBatchSize, numFrames, nbJoints, jointEmbeddingSize])
        
        let embeddedMotion = embeddedJointValues.reshaped(to: [origBatchSize, numFrames, nbJoints * jointEmbeddingSize])

        // squeeze all frames in a batch into first dimension
        let tmpMotionFrames = embeddedMotion.reshaped(to: [origBatchSize*numFrames, nbJoints*jointEmbeddingSize])
//        let tmpMotionFrames = motionPart.motion.reshaped(to: [origBatchSize*numFrames, nbJoints])

        let tmpMotionFeatures = motionDense(tmpMotionFrames) // 47 -> decoderDepth, or 47x5 -> decoderDepth
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

extension LangMotionCatDistTransformer {

    public init(config: LangMotionCatDistTransformerConfig,
                langEmbedding: Embedding<Float>, langPositionalEncoding: PositionalEncoding, encoder: Encoder,
                jointEmbedding: Embedding<Float>, motionDense: Dense<Float>, motionPositionalEncoding: PositionalEncoding, motionSegmentEmbedding: Embedding<Float>,
                motionNorm: LayerNorm<Float>, decoder: Decoder,
                catDistHead: MotionCatDistHead
    ) {
        self.config = config

        self.langEmbedding = langEmbedding
        self.langPositionalEncoding = langPositionalEncoding
        self.encoder = encoder

        self.jointEmbedding = jointEmbedding
        self.motionDense = motionDense
        self.motionPositionalEncoding = motionPositionalEncoding
        self.motionSegmentEmbedding = motionSegmentEmbedding
        self.motionNorm = motionNorm
        self.decoder = decoder

        self.catDistHead = catDistHead
    }
}
