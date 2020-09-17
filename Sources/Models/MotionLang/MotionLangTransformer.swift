import TensorFlow
import Datasets
import TranslationModels
import TextModels


// Transformer with MotionLangBatch

public struct MotionLangTransformer: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var positionalEncoding: PositionalEncoding
    public var motionNorm: LayerNorm<Float>
    public var motionDense: Dense<Float>
    public var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    public var generator: Generator
    @noDerivative public var modelSize: Int
    public init(inputSize: Int, targetVocabSize: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, headCount: Int = 8, dropoutProbability: Double = 0.1) {
        
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
        self.motionNorm = LayerNorm(featureCount: modelSize, axis: 2)
        self.motionDense = Dense<Float>(inputSize: inputSize, outputSize: modelSize)
        self.targetEmbed = Sequential(Embedding(vocabularySize: targetVocabSize, embeddingSize: modelSize,embeddingsInitializer: glorotUniform()), positionalEncoding)
        self.generator = Generator(dimModel: modelSize, vocabSize: targetVocabSize)
        self.modelSize = modelSize
    }
    
    @differentiable
    public func callAsFunction(_ input: MotionLangBatch) -> Tensor<Float> {
        let encodedMemory = self.encode(input: input)
        return self.decode(input: input, memory: encodedMemory)
    }
    
    @differentiable
    public func encode(input: MotionLangBatch) -> Tensor<Float> {
        let origBatchSize = input.motion.shape[0]
        let length = input.motion.shape[1]
        let numFrames = input.motion.shape[2]
        let hiddenSize = self.modelSize

        let tmpBatchSize = origBatchSize * length
        let tmpMotionFrames = input.motion.reshaped(to: [tmpBatchSize, numFrames])

        let tmpMotionFeatures = motionDense(tmpMotionFrames) // batch size here is origBatchSize*numFrames
        var motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, length, hiddenSize])
        motionFeatures = self.motionNorm(motionFeatures)

        motionFeatures = positionalEncoding(motionFeatures)

        let encoderInput: TransformerInput = TransformerInput(sequence: motionFeatures, attentionMask: input.mask, selfAttentionTemperature: 1.0)
        return self.encoder(encoderInput).lastLayerOutput
    }
    
    @differentiable
    public func decode(input: MotionLangBatch, memory: Tensor<Float>) -> Tensor<Float> {
        let embedded = self.targetEmbed(input.targetTokenIds)
        let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.mask, targetMask: input.targetMask, memory: memory, sourceAttentionTemperature: 1.0, selfAttentionTemperature: 1.0)
        return self.decoder(decoderInput).lastLayerOutput
    }
    
    @differentiable
    public func generate(input: MotionLangBatch) -> Tensor<Float> {
        return self.generator(self.callAsFunction(input))
    }
    @differentiable
    public func generate(input: Tensor<Float>) -> Tensor<Float> {
        self.generator(input)
    }
}
