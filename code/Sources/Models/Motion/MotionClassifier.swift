import Datasets
import ModelSupport
import TensorFlow
import TextModels
import ImageClassificationModels

public typealias FeatureTransformerEncoder = BERT
public typealias FeatureBatch = MotionBatch

public func createAttentionMask2(input2: Tensor<Float>, mask: Tensor<Int32>) -> Tensor<Float> {
//     let batchSize = text.tokenIds.shape[0]
    let batchSize = input2.shape[0]
//     let fromSequenceLength = text.tokenIds.shape[1]
//     let toSequenceLength = text.mask.shape[1]
    let fromSequenceLength = input2.shape[1]
    let toSequenceLength = input2.shape[1]
//     let reshapedMask = Tensor<Float>(text.mask.reshaped(to: [batchSize, 1, toSequenceLength]))
    let reshapedMask = Tensor<Float>(mask.reshaped(to: [batchSize, 1, toSequenceLength]))

    // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
    // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
//     let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1], on: text.mask.device)
    let broadcastOnes = Tensor<Float>(ones: [batchSize, fromSequenceLength, 1], on: mask.device)

    // We broadcast along two dimensions to create the mask.
    return broadcastOnes * reshapedMask
}

extension FeatureTransformerEncoder {
    @differentiable(wrt: self)
    public func callAsFunction(_ featureBatch: FeatureBatch) -> Tensor<Scalar> {
        print("ala ma kota")
        let input2 = featureBatch.motionFrames
        print("input2 = \(input2.shape)")
//         let tokenIds: Tensor<Int32> = Tensor<Int32>([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
//         let tokenTypeIds: Tensor<Int32> = Tensor<Int32>([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        let mask: Tensor<Int32> = Tensor<Int32>([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
//         let input: TextBatch = TextBatch(tokenIds: tokenIds, tokenTypeIds: tokenTypeIds, mask: mask)
//         let sequenceLength = input.tokenIds.shape[1]
        let sequenceLength = input2.shape[1]
//         let variant = withoutDerivative(at: self.variant)
        print(1)

        // Compute the input embeddings and apply layer normalization and dropout on them.
//         let tokenEmbeddings = tokenEmbedding(input.tokenIds)
        let tokenEmbeddings = input2

        print("tokenEmbeddings: \(tokenEmbeddings.shape)")
        
//         let tokenTypeEmbeddings = tokenTypeEmbedding(input.tokenTypeIds)
        let positionPaddingIndex: Int
        
        positionPaddingIndex = 0
        
        let positionEmbeddings = positionEmbedding.embeddings.slice(
            lowerBounds: [positionPaddingIndex, 0],
            upperBounds: [positionPaddingIndex + sequenceLength, -1]
        ).expandingShape(at: 0)
        
        
        print("positionEmbeddings: \(positionEmbeddings.shape)")
        
        var embeddings = tokenEmbeddings + positionEmbeddings

        // Add token type embeddings if needed, based on which BERT variant is being used.
//         embeddings = embeddings + tokenTypeEmbeddings

        embeddings = embeddingLayerNorm(embeddings)
        embeddings = embeddingDropout(embeddings)

        // TODO: do masking, but outside
        // TODO: get mask from 45th row (motion flag)
//         let mask = Tensor<Int32>([Int32](repeating: 1, count: sequenceLength))
        // Create an attention mask for the inputs with shape
        // `[batchSize, sequenceLength, sequenceLength]`.
//         let attentionMask = createAttentionMask(forTextBatch: input)
        let attentionMask = createAttentionMask2(input2: input2, mask: mask)

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
        // return Tensor<Float>([1.0, 1.0])
    }
}

public struct MotionClassifier: Module {
    public var featureExtractor: ResNet
    public var transformerEncoder: FeatureTransformerEncoder
    public var dense: Dense<Float>

    public init(featureExtractor: ResNet, transformerEncoder: FeatureTransformerEncoder, classCount: Int) {
        self.featureExtractor = featureExtractor
        self.transformerEncoder = transformerEncoder
        self.dense = Dense<Float>(inputSize: transformerEncoder.hiddenSize, outputSize: classCount)
    }

    func extractMotionFeatures(_ batchTensor: Tensor<Float>) -> Tensor<Float> {
        let tensorWidth = 60
        let stride = 10
        let tWidth = stride*2

        // sliding resnet feature extractor
        var t2: [Tensor<Float>] = []
        let origBatchSize = batchTensor.shape[0]
        let nElements = (tensorWidth/stride)-1
        for i in 0..<nElements {
            let start = i*stride
            let end = i*stride+tWidth
            // print(start, end)
            let t1 = batchTensor[0..., start..<end]
            // print(t1.shape)
            t2.append(t1)
        }
        let t3 = Tensor(concatenating: t2)
        // print(t3.shape)
        let emb2 = featureExtractor(t3)
        let outShape: Array<Int> = [origBatchSize, nElements, featureExtractor.classifier.weight.shape[1]]
        // print(outShape)
        let emb3 = emb2.reshaped(to: TensorShape(outShape))
        return emb3
    }

  /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: MotionBatch) -> Tensor<Float> {
        let motionFeatures = extractMotionFeatures(input.motionFrames)
        // FIXME: calculate feature mask
        let featureBatch = withoutDerivative(at:FeatureBatch(motionFrames: motionFeatures, motionFlag: input.motionFlag))
        return dense(transformerEncoder(featureBatch)[0..., 0])
    }
}
