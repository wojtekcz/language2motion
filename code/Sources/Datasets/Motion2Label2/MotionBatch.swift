import TensorFlow

public struct MotionBatch: Differentiable {
  public var motionFrames: Tensor<Float>
  /// Mask over the sequence of tokens specifying which ones are "real" as 
  /// opposed to "padding".
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  @noDerivative
  public let motionFlag: Tensor<Int32>

  @noDerivative
  public let origMotionFramesCount: Tensor<Int32>

  @differentiable
  public init(motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>) {
    self.motionFrames = motionFrames
    self.motionFlag = motionFlag
    self.origMotionFramesCount = origMotionFramesCount
  }
}

extension MotionBatch: Collatable {
  /// Creates an instance from collating `samples`.
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    self.init(
      motionFrames: .init(concatenating: samples.map({$0.motionFrames.expandingShape(at: 0)})), 
      motionFlag: .init(concatenating: samples.map({$0.motionFlag.expandingShape(at: 0)})),
      origMotionFramesCount: .init(concatenating: samples.map({$0.origMotionFramesCount.expandingShape(at: 0)}))
    )
  }
}


extension Collection where Element == MotionBatch {
  /// Returns the elements of `self`, padded to `maxLength` if specified
  /// or the maximum length of the elements in `self` otherwise.
  public func paddedAndCollated(to maxLength: Int? = nil) -> MotionBatch {
    if count == 1 { return first! }
    let maxLength = maxLength ?? self.map { $0.motionFrames.shape[1] }.max()!
    let paddedMotions = self.map { example -> MotionBatch in
      return MotionBatch(
        motionFrames: example.motionFrames.paddedOrCropped(to: maxLength),
        motionFlag: example.motionFlag.paddedOrCropped(to: maxLength),
        origMotionFramesCount: example.origMotionFramesCount)
    }
    return paddedMotions.collated
  }
}
