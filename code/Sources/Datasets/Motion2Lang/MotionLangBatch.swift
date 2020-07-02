import Foundation
import TensorFlow

// TODO: add targetTokenCount and sampleID
public struct MotionLangBatch: KeyPathIterable {
    public var motionFrames: Tensor<Float>
    public let origMotionFramesCount: Tensor<Int32>
    /// IDs that correspond to the vocabulary used while tokenizing.
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    // aka src
    
    public var targetTokenIds: Tensor<Int32>
    // aka tgt
    
    /// IDs of the token types (e.g., sentence A and sentence B in BERT).
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    // public var tokenTypeIds: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.
    
    /// Mask over the sequence of tokens specifying which ones are "real" as opposed to "padding".
    /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
    public var mask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetMask: Tensor<Float> // TODO: !!! Mutable in order to allow for batching.
    
    public var targetTruth: Tensor<Int32>
    
    // public var tokenCount: Int32
    
    public init(motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>, target: Tensor<Int32>, targetPadId: Int32) {
        self.motionFrames = motionFrames
        self.origMotionFramesCount = origMotionFramesCount

        let mask = Tensor<Float>(Tensor(zerosLike: motionFlag)
            .replacing(with: Tensor(onesLike: motionFlag), where: motionFlag .!= Tensor.init(0))
            .expandingShape(at: 1))
        self.mask = mask

        let rangeExceptLast = 0..<(target.shape[1] - 1)
        self.targetTokenIds = target[0...,rangeExceptLast]
        self.targetTruth = target[0..., 1...]
        self.targetMask = MotionLangBatch.makeStandardMask(target: self.targetTokenIds, pad: targetPadId)
    }

    public init(motionFrames: Tensor<Float>, mask: Tensor<Float>,  origMotionFramesCount: Tensor<Int32>, targetTokenIds: Tensor<Int32>, targetMask: Tensor<Float>, targetTruth: Tensor<Int32>) {
        self.motionFrames = motionFrames
        self.mask = mask
        self.origMotionFramesCount = origMotionFramesCount
        self.targetTokenIds = targetTokenIds
        self.targetMask = targetMask
        self.targetTruth = targetTruth
    }

    static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask2(size: target.shape.last!)
        return Tensor<Float>(targetMask)
    }
}

public func subsequentMask2(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}

extension MotionLangBatch {
    public init(copying batch: MotionLangBatch, to device: Device) {
        self.motionFrames = Tensor<Float>(copying: batch.motionFrames, to: device)
        self.mask = Tensor<Float>(copying: batch.mask, to: device)
        self.origMotionFramesCount = Tensor<Int32>(copying: batch.origMotionFramesCount, to: device)
        self.targetTokenIds = Tensor<Int32>(copying: batch.targetTokenIds, to: device)
        self.targetMask = Tensor<Float>(copying: batch.targetMask, to: device)
        self.targetTruth = Tensor<Int32>(copying: batch.targetTruth, to: device)
    }
}
