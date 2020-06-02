import TensorFlow

public struct MotionBatch {
  public let motionFrames: Tensor<Float>
  /// Mask over the sequence of tokens specifying which ones are "real" as 
  /// opposed to "padding".
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  public let motionFlag: Tensor<Int32>

  public init(
    motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>
  ) {
    self.motionFrames = motionFrames
    self.motionFlag = motionFlag
  }
}

extension MotionBatch: Collatable {
  /// Creates an instance from collating `samples`.
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    self.init(
      motionFrames: .init(concatenating: samples.map(\.motionFrames)), 
      motionFlag: .init(concatenating: samples.map(\.motionFlag))
    )
  }
}


extension Collection where Element == MotionBatch {
  /// Returns the elements of `self`, padded to `maxLength` if specified
  /// or the maximum length of the elements in `self` otherwise.
  public func paddedAndCollated(to maxLength: Int? = nil) -> MotionBatch {
    if count == 1 { return first! }
    let maxLength = maxLength ?? self.map { $0.motionFrames.shape[1] }.max()!
    let paddedMotions = self.map { text -> MotionBatch in
      return MotionBatch(
        motionFrames: text.motionFrames.paddedOrCropped(to: maxLength),
        motionFlag: text.motionFlag.paddedOrCropped(to: maxLength))
    }
    return paddedMotions.collated
  }
}
