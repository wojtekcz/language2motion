import TensorFlow
import Datasets
import TextModels
import TranslationModels


// Transformer with LangMotionBatch

public struct LangMotionTransformerOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var preds: MixtureModelPreds
    public var encoded: Tensor<Scalar>
    public var decoded: DecoderOutput<Scalar>

    @differentiable
    public init(preds: MixtureModelPreds, encoded: Tensor<Scalar>, decoded: DecoderOutput<Scalar>) {
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
    public var motionDense: Dense<Float>
    public var contextDense: Dense<Float>
    public var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding> 
    public var mixtureModel: MotionGaussianMixtureModel
    @noDerivative public var modelSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int
    @noDerivative public var doMotionDense: Bool
    @noDerivative public var contextSize: Int
    @noDerivative public var motionPositionalEncodingSize: Int

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
        let motionPositionalEncodingSize = 32
        self.motionPositionalEncoding = PositionalEncoding(size: motionPositionalEncodingSize, dropoutProbability: dropoutProbability, maxLength: motionMaxPositionalLength)
        self.embedding = Embedding<Float>(vocabularySize: vocabSize, embeddingSize: modelSize, embeddingsInitializer: glorotUniform())
        self.sourceEmbed = Sequential(embedding, positionalEncoding)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.motionDense = Dense<Float>(inputSize: nbJoints, outputSize: modelSize)
        let contextSize = 128
        self.contextDense = Dense<Float>(inputSize: modelSize, outputSize: 128) // FIXME: parametrize contextSize = 128
        self.mixtureModel = MotionGaussianMixtureModel(inputSize: modelSize*layerCount, nbJoints: nbJoints, nbMixtures: nbMixtures)
        self.modelSize = modelSize
        self.nbJoints = nbJoints        
        self.nbMixtures = nbMixtures
        self.doMotionDense = doMotionDense
        self.contextSize = contextSize
        self.motionPositionalEncodingSize = motionPositionalEncodingSize
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch.Source) -> LangMotionTransformerOutput<Float> {
        let encodedMemory = self.encode(input: input.sentence)
        let decoded = self.decode(sourceMask: input.sentence.mask, motionPart: input.motionPart, memory: encodedMemory)
        // reformat decoded.allOutputs[] into one tensor
        let mixtureModelInput = Tensor<Float>(concatenating: decoded.allOutputs, alongAxis: 2)
        return LangMotionTransformerOutput(preds: self.mixtureModel(mixtureModelInput), encoded: encodedMemory, decoded: decoded)
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
            // TODO: kill motionDense layer eventually
            let shape = motionPart.motion.shape
            let (origBatchSize, numFrames) = (shape[0], shape[1])

            let tmpBatchSize = origBatchSize * numFrames
            let tmpMotionPart = motionPart.motion.reshaped(to: [tmpBatchSize, nbJoints])

            // FIXME: make targetEmbed() work
            let tmpMotionPartFeatures = motionDense(tmpMotionPart) // batch size here is origBatchSize*numFrames
            motionPartFeatures = tmpMotionPartFeatures.reshaped(to: [origBatchSize, numFrames, self.modelSize])
            motionPartFeatures = motionPositionalEncoding(motionPartFeatures)
        } else {
            // TODO: refactor this out
            // assuming modelSize = 256
            let shape = motionPart.motion.shape
            let (batchSize, numFrames) = (shape[0], shape[1])

            // motion positional encoding
            var motionPositionalEncodingVector = Tensor<Float>(repeating: 0.0, shape: [batchSize, numFrames, motionPositionalEncodingSize])
            motionPositionalEncodingVector = motionPositionalEncoding(motionPositionalEncodingVector)
            
            // current motion
            let currentMotion = motionPart.motion

            // compute contextVector
            let numTokens = memory.shape[1]
            let mask = sourceMask[0..., 0, 0...].expandingShape(at: 2).broadcasted(to: [batchSize, numTokens, modelSize])
            let maskedMemory = memory * mask
            let meanMemory = maskedMemory.mean(alongAxes: 1).squeezingShape(at: 1) // get mean across steps

            let contextVector = contextDense(meanMemory).expandingShape(at: 1).broadcasted(to: [batchSize, numFrames, contextSize])

            // previousMotion
            let previousMotion = motionPart.previousMotion

            // compute padding
            let motionFramePadding = Tensor<Float>(repeating: 0.0, shape: [batchSize, numFrames, modelSize - (1+motionPositionalEncodingSize+nbJoints*2+contextSize)])

            let tensorStack = [motionPart.startFlag, motionPositionalEncodingVector, currentMotion, previousMotion, contextVector, motionFramePadding]
            let tmpMotionPartFeatures = Tensor<Float>(concatenating: tensorStack, alongAxis: 2)

            // FIXME: preserve following?
            // tile motion along joints dimension
            // let multiplyBy = modelSize/nbJoints+1
            // let tmpMotionPartFeatures = motionPart.motion.tiled(multiples: [1, 1, multiplyBy])[0..., 0..., 0..<modelSize]
            // motionPartFeatures = motionPositionalEncoding(tmpMotionPartFeatures)
            motionPartFeatures = tmpMotionPartFeatures
        }
        let decoderInput = DecoderInput(sequence: motionPartFeatures, sourceMask: sourceMask, targetMask: motionPart.mask, memory: memory)
        return self.decoder(decoderInput)
    }
}

extension LangMotionTransformer {

    public init(encoder: Encoder, decoder: Decoder, embedding: Embedding<Float>, positionalEncoding: PositionalEncoding, motionPositionalEncoding: PositionalEncoding, 
        motionDense: Dense<Float>, contextDense: Dense<Float>, sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>, 
        mixtureModel: MotionGaussianMixtureModel, modelSize: Int, nbJoints: Int, nbMixtures: Int, doMotionDense: Bool) 
    {
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.positionalEncoding = positionalEncoding
        self.motionPositionalEncoding = motionPositionalEncoding
        self.motionDense = motionDense
        self.contextDense = contextDense
        self.sourceEmbed = sourceEmbed
        self.mixtureModel = mixtureModel
        self.modelSize = modelSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.doMotionDense = doMotionDense
        self.contextSize = 128 // FIXME: parametrize contextSize
        self.motionPositionalEncodingSize = 32 // FIXME: parametrize motionPositionalEncodingSize
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
