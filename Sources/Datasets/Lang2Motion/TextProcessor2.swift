import Foundation
import TensorFlow
import ModelSupport

public struct TextProcessor2 {

    public init(vocabulary: Vocabulary, tokenizer: Tokenizer, maxSequenceLength: Int? = nil) {
    }

    public func preprocess(example: Lang2Motion.Example) -> LangMotionBatch {
        // let singleBatch = MotionLangBatch(source: sourceTensor, target: targetTensor, sourcePadId: padId, targetPadId: padId)
        //motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>, target: Tensor<Int32>, targetPadId: Int32
        // let motionFrames: Tensor<Float> = Tensor([[1, 2, 3]])
        // let motionFlag: Tensor<Int32> = Tensor([[1, 2, 3]])
        // let origMotionFramesCount: Tensor<Int32> = Tensor([1])
        let singleBatch = LangMotionBatch() // motionFrames: motionFrames, motionFlag: motionFlag,  origMotionFramesCount: origMotionFramesCount, target: targetTensor, targetPadId: padId)
        return singleBatch
    }
}
