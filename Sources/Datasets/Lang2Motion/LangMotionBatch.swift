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
    public var targetTruth: Tensor<Float>
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
}
