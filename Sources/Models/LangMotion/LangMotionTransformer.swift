import TensorFlow
import Datasets
import TextModels
import TranslationModels


// Transformer with LangMotionBatch

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
    public var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding> 
    public var mixtureModel: MotionGaussianMixtureModel
    @noDerivative public var modelSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int
    @noDerivative public var motionPositionalEncodingSize: Int

    public init(vocabSize: Int, nbJoints: Int, nbMixtures: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, 
                headCount: Int = 8, dropoutProbability: Double = 0.1, sentenceMaxPositionalLength: Int = 5000, motionMaxPositionalLength: Int = 5000) {
        
        let attention = MultiHeadAttention(sourceSize: modelSize,
                                           targetSize: modelSize,
                                           headCount: headCount,
                                           headSize: modelSize/headCount,
                                           matrixResult: false)
        let feedForward = PositionwiseFeedForward(dimensionalityModel: modelSize,
                                                  innerLayerDimensionality: feedForwardSize)
        
        self.positionalEncoding = PositionalEncoding(size: modelSize, dropoutProbability: dropoutProbability, maxLength: sentenceMaxPositionalLength)
        let motionPositionalEncodingSize = 32
        self.motionPositionalEncoding = PositionalEncoding(size: motionPositionalEncodingSize, dropoutProbability: dropoutProbability, maxLength: motionMaxPositionalLength)
        self.embedding = Embedding<Float>(vocabularySize: vocabSize, embeddingSize: modelSize, embeddingsInitializer: glorotUniform())
        self.sourceEmbed = Sequential(embedding, positionalEncoding)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.motionNorm = LayerNorm(featureCount: modelSize, axis: 2)
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: modelSize*layerCount, nbJoints: nbJoints, nbMixtures: nbMixtures)
        self.modelSize = modelSize
        self.nbJoints = nbJoints        
        self.nbMixtures = nbMixtures
        self.motionPositionalEncodingSize = motionPositionalEncodingSize
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> LangMotionTransformerOutput<Float> {
        let encoded = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        // reformat decoded.allOutputs[] into one tensor
        let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
        return LangMotionTransformerOutput(preds: self.mixtureModel(mixtureModelInput), encoded: encoded, decoded: decoded)
    }
    
    @differentiable
    public func encode(input: LangMotionBatch.Sentence) -> EncoderOutput<Float> {
        let embedded = self.sourceEmbed(input.tokenIds)
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(sourceMask: Tensor<Float>, motionPart: LangMotionBatch.MotionPart, memory: Tensor<Float>) -> DecoderOutput<Float> {
        var motionPartFeatures: Tensor<Float>

        // start flag, pos enc, current motion, padding with motion
        let shape = motionPart.motion.shape
        let (batchSize, numFrames) = (shape[0], shape[1])

        // motion positional encoding
        var motionPositionalEncodingVector = Tensor<Float>(repeating: 0.0, shape: [batchSize, numFrames, motionPositionalEncodingSize])
        motionPositionalEncodingVector = motionPositionalEncoding(motionPositionalEncodingVector)
        
        // compute padding
        let paddingSize = modelSize - (1 + motionPositionalEncodingSize + nbJoints)
        
        let multiplyBy = paddingSize/nbJoints + 1
        let motionFramePadding = motionPart.motion.tiled(multiples: [1, 1, multiplyBy])[0..., 0..., 0..<paddingSize]

        // stack everything together
        let tensorStack = [motionPart.startFlag, motionPositionalEncodingVector, motionPart.motion, motionFramePadding]
        let tmpMotionPartFeatures = Tensor<Float>(concatenating: tensorStack, alongAxis: 2)
        motionPartFeatures = tmpMotionPartFeatures

        motionPartFeatures = self.motionNorm(motionPartFeatures)

        let decoderInput = DecoderInput(sequence: motionPartFeatures, sourceMask: sourceMask, targetMask: motionPart.mask, memory: memory)
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>, positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding, 
        sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>, 
        mixtureModel: MotionGaussianMixtureModel, modelSize: Int, nbJoints: Int, nbMixtures: Int, motionNorm: LayerNorm<Float>) 
    {
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.sourceEmbed = sourceEmbed
        self.mixtureModel = mixtureModel
        self.modelSize = modelSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.motionPositionalEncodingSize = 32 // FIXME: parametrize motionPositionalEncodingSize
        self.motionNorm = motionNorm
    }

    public init(config: LangMotionTransformerConfig) {
        self.init(
            vocabSize: config.vocabSize, 
            nbJoints: config.nbJoints, 
            nbMixtures: config.nbMixtures, 
            layerCount: config.layerCount, 
            modelSize: config.modelSize, 
            feedForwardSize: config.feedForwardSize, 
            headCount: config.headCount, 
            dropoutProbability: config.dropoutProbability, 
            sentenceMaxPositionalLength: config.sentenceMaxPositionalLength, 
            motionMaxPositionalLength: config.motionMaxPositionalLength
        )
    }
}
