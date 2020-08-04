import Datasets
import ModelSupport
import TensorFlow
import TextModels
import ImageClassificationModels


public struct ResNetMotionClassifier: Module, MotionClassifierProtocol {
    public var resNetClassifier: ResNet

    @noDerivative
    public let maxSequenceLength: Int

    public init(resNetClassifier: ResNet, maxSequenceLength: Int) {
        self.resNetClassifier = resNetClassifier
        self.maxSequenceLength = maxSequenceLength
    }

    /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: MotionBatch) -> Tensor<Float> {
        return resNetClassifier(input.motionFrames.expandingShape(at: 3))
    }
}
