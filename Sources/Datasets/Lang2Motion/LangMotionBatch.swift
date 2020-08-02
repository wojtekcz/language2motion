import Foundation
import TensorFlow

public struct LangMotionBatch: KeyPathIterable {
    public let sampleID: Tensor<Int32>

    // source
    public var tokenIds: Tensor<Int32>
    public var mask: Tensor<Float>
    public let tokenCount: Tensor<Int32>
    
    // target
    public var targetMotionFrames: Tensor<Float>
    public var targetMask: Tensor<Float>
    public var targetTruth: Tensor<Float> // TODO: target truth should be a STRUCT with motion and stop components
    // TODO: how targetMask is used and consumed?
    // used by transformer decoder? by mixture model?
    public let origMotionFramesCount: Tensor<Int32>

    public init(sampleID: Tensor<Int32>, 
                tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>, 
                targetMotionFrames: Tensor<Float>, targetMask: Tensor<Float>,
                targetTruth: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
        self.sampleID = sampleID

        self.tokenIds = tokenIds
        self.mask = mask
        self.tokenCount = tokenCount

        self.targetMotionFrames = targetMotionFrames
        self.targetMask = targetMask
        self.targetTruth = targetTruth
        self.origMotionFramesCount = origMotionFramesCount
    }

    public static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask3(size: target.shape.last!)
        return Tensor<Float>(targetMask)
    }
}

public func subsequentMask3(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}

extension LangMotionBatch {
    public init(copying batch: LangMotionBatch, to device: Device) {
        self.sampleID = Tensor<Int32>(copying: batch.sampleID, to: device)

        self.tokenIds = Tensor<Int32>(copying: batch.tokenIds, to: device)
        self.mask = Tensor<Float>(copying: batch.mask, to: device)
        self.tokenCount = Tensor<Int32>(copying: batch.tokenCount, to: device)

        self.targetMotionFrames = Tensor<Float>(copying: batch.targetMotionFrames, to: device)
        self.targetMask = Tensor<Float>(copying: batch.targetMask, to: device)
        self.targetTruth = Tensor<Float>(copying: batch.targetTruth, to: device)
        self.origMotionFramesCount = Tensor<Int32>(copying: batch.origMotionFramesCount, to: device)
    }
}
