import Datasets
import ModelSupport
import TensorFlow
import TextModels
import ImageClassificationModels

public typealias FeatureTransformerEncoder = BERT
public typealias FeatureBatch = MotionBatch

extension FeatureTransformerEncoder {
    @differentiable(wrt: self)
    public func callAsFunction(_ featureBatch: FeatureBatch) -> Tensor<Scalar> {
        let sequenceLength = featureBatch.motionFrames.shape[1]
        let positionPaddingIndex: Int
        
        positionPaddingIndex = 0
        
        let positionEmbeddings = positionEmbedding.embeddings.slice(
            lowerBounds: [positionPaddingIndex, 0],
            upperBounds: [positionPaddingIndex + sequenceLength, -1]
        ).expandingShape(at: 0)

        var embeddings = featureBatch.motionFrames + positionEmbeddings
        embeddings = embeddingLayerNorm(embeddings)
        embeddings = embeddingDropout(embeddings)

        // Create an attention mask for the inputs with shape
        // `[batchSize, sequenceLength, sequenceLength]`.
        let attentionMask = FeatureTransformerEncoder.createAttentionMask(forFeatureBatch: featureBatch)

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        var transformerInput = embeddings.reshapedToMatrix()
        let batchSize = embeddings.shape[0]

        // Run the stacked transformer.
        for layerIndex in 0..<(withoutDerivative(at: encoderLayers) { $0.count }) {
            transformerInput = encoderLayers[layerIndex](TransformerInput(
            sequence: transformerInput,
            attentionMask: attentionMask,
            batchSize: batchSize))
        }

        // Reshape back to the original tensor shape.
        return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)
    }

    static public func createAttentionMask(forFeatureBatch featureBatch: FeatureBatch) -> Tensor<Float> {
        let batchSize = featureBatch.motionFrames.shape[0]
        let mask = featureBatch.motionFlag
        // let fromSequenceLength = text.tokenIds.shape[1]
        // let toSequenceLength = text.mask.shape[1]
        let fromSequenceLength = featureBatch.motionFrames.shape[1]
        let toSequenceLength = featureBatch.motionFrames.shape[1]
        let reshapedMask = Tensor<Float>(mask.reshaped(to: [batchSize, 1, toSequenceLength]))

        // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
        // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
        let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1], on: mask.device)

        // We broadcast along two dimensions to create the mask.
        return broadcastOnes * reshapedMask
    }
}

public struct Prediction {
    public let classIdx: Int
    public let className: String
    public let probability: Float
}

public struct DenseMotionClassifier: Module, Regularizable {
    public var featureExtractor: Dense<Float>
    public var transformerEncoder: FeatureTransformerEncoder
    public var dense: Dense<Float>

    @noDerivative
    public let maxSequenceLength: Int

    public var regularizationValue: TangentVector {
        TangentVector(
        featureExtractor: featureExtractor.regularizationValue,
        transformerEncoder: transformerEncoder.regularizationValue,
        dense: dense.regularizationValue)
    }

    public init(transformerEncoder: FeatureTransformerEncoder, inputSize: Int, classCount: Int, maxSequenceLength: Int) {
        self.featureExtractor = Dense<Float>(inputSize: inputSize, outputSize: transformerEncoder.hiddenSize)
        self.transformerEncoder = transformerEncoder
        self.dense = Dense<Float>(inputSize: transformerEncoder.hiddenSize, outputSize: classCount)
        self.maxSequenceLength = maxSequenceLength
    }

    @differentiable(wrt: self)
    func extractMotionFeatures(_ input: MotionBatch) -> FeatureBatch {
        let origBatchSize = input.motionFrames.shape[0]
        let length = input.motionFrames.shape[1]
        let numFeatures = input.motionFrames.shape[2] //numFrames
        let hiddenSize = transformerEncoder.hiddenSize

        let tmpBatchSize = origBatchSize * length
        let tmpMotionFrames = input.motionFrames.reshaped(to: [tmpBatchSize, numFeatures])

        let tmpMotionFeatures = featureExtractor(tmpMotionFrames) // batch size here is origBatchSize*numFeatures
        let motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, length, hiddenSize])

        return FeatureBatch(motionFrames: motionFeatures, motionFlag: input.motionFlag, origMotionFramesCount: input.origMotionFramesCount)
    }

    /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: MotionBatch) -> Tensor<Float> {
        let featureBatch = extractMotionFeatures(input)
        let transformerEncodings = transformerEncoder(featureBatch)
        let classifierOutput = dense(transformerEncodings[0..., 0])
        return classifierOutput
    }

    public func predict(motionSamples: [MotionSample], labels: [String], batchSize: Int = 10) -> [Prediction] {
        Context.local.learningPhase = .inference
        let validationExamples = motionSamples.map {
            (example) -> MotionBatch in
            let motionFrames = Tensor<Float>(example.motionFramesArray)
            let mfIdx = MotionFrame.cjpMotionFlagIdx
            let motionFlag = Tensor<Int32>(motionFrames[0..., mfIdx...mfIdx].squeezingShape(at: 1))
            let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))
            let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag, origMotionFramesCount: origMotionFramesCount)
            return motionBatch
        }

        let validationBatches = validationExamples.inBatches(of: batchSize).map { 
            $0.paddedAndCollated(to: maxSequenceLength)
        }

        var preds: [Prediction] = []
        for batch in validationBatches {
            let logits = self(batch)
            let probs = softmax(logits, alongAxis: 1)
            let classIdxs = logits.argmax(squeezingAxis: 1)
            let batchPreds = (0..<classIdxs.shape[0]).map { 
                (idx) -> Prediction in
                let classIdx: Int = Int(classIdxs[idx].scalar!)
                let prob = probs[idx, classIdx].scalar!
                return Prediction(classIdx: classIdx, className: labels[classIdx], probability: prob)
            }
            preds.append(contentsOf: batchPreds)
        }
        return preds
    }
}
