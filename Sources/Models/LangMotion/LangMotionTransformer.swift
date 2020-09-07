import TensorFlow
import Datasets
import TextModels
import TranslationModels


// Transformer with LangMotionBatch2

public struct LangMotionTransformer: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var embedding: Embedding<Float>
    public var positionalEncoding: PositionalEncoding
    public var motionPositionalEncoding: PositionalEncoding
    public var motionDense: Dense<Float>
    public var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding> 
    public var mixtureModel: MotionGaussianMixtureModel
    @noDerivative public var modelSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int
    @noDerivative public var doMotionDense: Bool

    public init(vocabSize: Int, nbJoints: Int, nbMixtures: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, 
                headCount: Int = 8, dropoutProbability: Double = 0.1, sentenceMaxPositionalLength: Int = 5000, motionMaxPositionalLength: Int = 5000, doMotionDense: Bool = true) {
        
        let attention = MultiHeadAttention(sourceSize: modelSize,
                                           targetSize: modelSize,
                                           headCount: headCount,
                                           headSize: modelSize/headCount,
                                           matrixResult: false)
        let feedForward = PositionwiseFeedForward(dimensionalityModel: modelSize,
                                                  innerLayerDimensionality: feedForwardSize)
        
        self.positionalEncoding = PositionalEncoding(size: modelSize, dropoutProbability: dropoutProbability, maxLength: sentenceMaxPositionalLength)
        self.motionPositionalEncoding = PositionalEncoding(size: modelSize, dropoutProbability: dropoutProbability, maxLength: motionMaxPositionalLength)
        self.embedding = Embedding<Float>(vocabularySize: vocabSize, embeddingSize: modelSize, embeddingsInitializer: glorotUniform())
        self.sourceEmbed = Sequential(embedding, positionalEncoding)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.motionDense = Dense<Float>(inputSize: nbJoints, outputSize: modelSize)
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: modelSize*layerCount, nbJoints: nbJoints, nbMixtures: nbMixtures)
        self.modelSize = modelSize
        self.nbJoints = nbJoints        
        self.nbMixtures = nbMixtures
        self.doMotionDense = doMotionDense
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> MixtureModelPreds {
        let encodedMemory = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sentence.mask, motionPart: input.motionPart, memory: encodedMemory)
        // reformat decoded.allOutputs[] into one tensor
        let mixtureModelInput = Tensor<Float>(concatenating: decoded.allOutputs, alongAxis: 2)
        return self.mixtureModel(mixtureModelInput)
    }
    
    @differentiable
    public func encode(input: LangMotionBatch.Sentence) -> Tensor<Float> {
        let embedded = self.sourceEmbed(input.tokenIds)
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(sourceMask: Tensor<Float>, motionPart: LangMotionBatch.MotionPart, memory: Tensor<Float>) -> DecoderOutput<Float> {
        var motionPartFeatures: Tensor<Float>
        if doMotionDense {
            let shape = motionPart.motion.shape
            let (origBatchSize, numFrames) = (shape[0], shape[1])

            let tmpBatchSize = origBatchSize * numFrames
            let tmpMotionPart = motionPart.motion.reshaped(to: [tmpBatchSize, nbJoints])

            // FIXME: make targetEmbed() work
            let tmpMotionPartFeatures = motionDense(tmpMotionPart) // batch size here is origBatchSize*numFrames
            motionPartFeatures = tmpMotionPartFeatures.reshaped(to: [origBatchSize, numFrames, self.modelSize])
            motionPartFeatures = motionPositionalEncoding(motionPartFeatures)
        } else {
            // tile motion along joints dimension
            let multiplyBy = modelSize/nbJoints+1
            let tmpMotionPartFeatures = motionPart.motion.tiled(multiples: [1, 1, multiplyBy])[0..., 0..., 0..<modelSize]
            motionPartFeatures = motionPositionalEncoding(tmpMotionPartFeatures)
        }
        let decoderInput = DecoderInput(sequence: motionPartFeatures, sourceMask: sourceMask, targetMask: motionPart.mask, memory: memory)
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>, positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding, 
        motionDense: Dense<Float>, sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>, 
        mixtureModel: MotionGaussianMixtureModel, modelSize: Int, nbJoints: Int, nbMixtures: Int, doMotionDense: Bool) 
    {
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.motionDense = motionDense
        self.sourceEmbed = sourceEmbed
        self.mixtureModel = mixtureModel
        self.modelSize = modelSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.doMotionDense = doMotionDense
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
            motionMaxPositionalLength: config.motionMaxPositionalLength,
            doMotionDense: config.doMotionDense
        )
    }
}
