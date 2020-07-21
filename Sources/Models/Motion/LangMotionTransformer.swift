import TensorFlow
import Datasets
import TranslationModels


// Transformer with LangMotionBatch

public struct LangMotionTransformer: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var positionalEncoding: PositionalEncoding
    public var motionDense: Dense<Float>
    // public var sourceEmbed: Sequential<Dense<Float>, PositionalEncoding> 
    public var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding> 
    // public var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding> // kill it
    public var generator: Generator
    @noDerivative public var modelSize: Int

    public init(vocabSize: Int, nbJoints: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, headCount: Int = 8, dropoutProbability: Double = 0.1) {
        
        let attention = MultiHeadAttention(sourceSize: modelSize,
                                           targetSize: modelSize,
                                           headCount: headCount,
                                           headSize: modelSize/headCount,
                                           matrixResult: false)
        let feedForward = PositionwiseFeedForward(dimensionalityModel: modelSize,
                                                  innerLayerDimensionality: feedForwardSize)
        
        positionalEncoding = PositionalEncoding(size: modelSize,
                                                    dropoutProbability: dropoutProbability)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        motionDense = Dense<Float>(inputSize: nbJoints, outputSize: modelSize)
        // self.motionEmbed = Dense<Float>(inputSize: inputSize, outputSize: modelSize)
        self.sourceEmbed = Sequential(Embedding(vocabularySize: vocabSize, embeddingSize: modelSize, embeddingsInitializer: glorotUniform()), positionalEncoding)
        // self.sourceEmbed = Sequential(motionDense, positionalEncoding)
        // self.targetEmbed = Sequential(Embedding(vocabularySize: targetVocabSize, embeddingSize: modelSize,embeddingsInitializer: glorotUniform()), positionalEncoding)
        self.generator = Generator(dimModel: modelSize, vocabSize: nbJoints)
        self.modelSize = modelSize
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch) -> Tensor<Float> {
        let encodedMemory = self.encode(input: input)
        return self.decode(input: input, memory: encodedMemory)
    }
    
    @differentiable
    public func encode(input: LangMotionBatch) -> Tensor<Float> {
        let embedded = self.sourceEmbed(input.tokenIds)
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(input: LangMotionBatch, memory: Tensor<Float>) -> Tensor<Float> {
        let shape = input.targetMotionFrames.shape
        let origBatchSize = shape[0]
        let numFrames = shape[1]
        let numFeatures = shape[2]

        let tmpBatchSize = origBatchSize * numFrames
        let tmpMotionFrames = input.targetMotionFrames.reshaped(to: [tmpBatchSize, numFeatures])

        // FIXME: make targetEmbed() work
        let tmpMotionFeatures = motionDense(tmpMotionFrames) // batch size here is origBatchSize*numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, numFrames, self.modelSize])
        motionFeatures = positionalEncoding(motionFeatures)

        let decoderInput = DecoderInput(sequence: motionFeatures, sourceMask: input.mask, targetMask: input.targetMask, memory: memory)
        return self.decoder(decoderInput)
    }
    
    @differentiable
    public func generate(input: LangMotionBatch) -> Tensor<Float> {
        return self.generator(self.callAsFunction(input))
    }
    @differentiable
    public func generate(input: Tensor<Float>) -> Tensor<Float> {
        self.generator(input)
    }
}
