import Datasets
import ModelSupport
import TensorFlow
import TextModels
import ImageClassificationModels

public typealias FeatureTransformerEncoder = BERT
public typealias FeatureBatch = MotionBatch

public func createAttentionMask(forFeatureBatch featureBatch: FeatureBatch) -> Tensor<Float> {
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
        let attentionMask = createAttentionMask(forFeatureBatch: featureBatch)

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
}

public struct MotionClassifier: Module {
    public var featureExtractor: ResNet
    public var transformerEncoder: FeatureTransformerEncoder
    public var dense: Dense<Float>

    @noDerivative
    public let maxSequenceLength: Int

    public init(featureExtractor: ResNet, transformerEncoder: FeatureTransformerEncoder, classCount: Int, maxSequenceLength: Int) {
        self.featureExtractor = featureExtractor
        self.transformerEncoder = transformerEncoder
        self.dense = Dense<Float>(inputSize: transformerEncoder.hiddenSize, outputSize: classCount)
        self.maxSequenceLength = maxSequenceLength
    }

    @differentiable(wrt: self)
    func extractMotionFeatures(_ input: MotionBatch) -> FeatureBatch {
        /// sliding 1-channel ResNet feature extractor
        let stride = 10
        let sliceWidth = stride*2 // 20
        let numFeatures = (maxSequenceLength/stride)-1
        let origBatchSize = input.motionFrames.shape[0]
        let hiddenSize = featureExtractor.classifier.weight.shape[1]

        // create set of slices
        var tmpMotionFrameSlices: [Tensor<Float>] = []
        var tmpMaskSlices: [Tensor<Int32>] = []
        let tmpMotionFrames = input.motionFrames.expandingShape(at: 3)

        for i in 0..<numFeatures {
            let start = i*stride
            let end = i*stride+sliceWidth
            let aSlice = tmpMotionFrames[0..., start..<end]
            tmpMotionFrameSlices.append(aSlice)
            // create mask from motion flag slices that contain at least one non-zero value
            let motionFlagSlice = input.motionFlag[0..., start..<end]
            let motionMaskSlice = motionFlagSlice.max(alongAxes: 1)
            tmpMaskSlices.append(motionMaskSlice)
        }
        let motionFrameSlices = Tensor(concatenating: tmpMotionFrameSlices)
        let tmpMotionFeatures = featureExtractor(motionFrameSlices) // batch size here is origBatchSize*numFeatures
        let motionFeatures = tmpMotionFeatures.reshaped(to: [origBatchSize, numFeatures, hiddenSize])

        let mask = Tensor(concatenating: tmpMaskSlices).reshaped(to: [origBatchSize, numFeatures])
        return FeatureBatch(motionFrames: motionFeatures, motionFlag: mask, origMotionFramesCount: input.origMotionFramesCount)
    }

    /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: MotionBatch) -> Tensor<Float> {
        // print("MotionClassifier.callAsFunction()")
        // print("  input.motionFrames.shape: \(input.motionFrames.shape)")
        // print("  input.motionFlag.shape: \(input.motionFlag.shape)")
        // print("  input.origMotionFramesCount: \(input.origMotionFramesCount)")

        let featureBatch = extractMotionFeatures(input)
        // print("  featureBatch.motionFrames.shape: \(featureBatch.motionFrames.shape)")
        // print("  featureBatch.motionFlag.shape: \(featureBatch.motionFlag.shape)")

        let transformerEncodings = transformerEncoder(featureBatch)
        // print("  transformerEncodings.shape: \(transformerEncodings.shape)")
        // print("MotionClassifier.callAsFunction() - stop")
        let classifierOutput = dense(transformerEncodings[0..., 0])
        return classifierOutput
    }
}
