import Foundation
import TensorFlow
import Datasets

public class MotionLangDecoder {
    public static func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch.MLSource, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
        let memory = model.encode(input: input).lastLayerOutput
        var ys = Tensor(repeating: startSymbol, shape: [1,1])
        for _ in 0..<maxLength {
            let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
            var motionPartMask = MotionLangBatch.makeStandardMask(target: motionPartFlag, pad: 0, shiftRight: true)
            let motionLen = Int(motionPartFlag.sum().scalar!)
            motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
            motionPartMask = abs(motionPartMask)

            let decoderInput = MotionLangBatch.MLSource(sampleID: input.sampleID, motion: input.motion,
                                         mask: input.mask,
                                         origMotionFramesCount: input.origMotionFramesCount,
                                         targetTokenIds: ys,
                                         targetMask: motionPartMask
                                         )
            let out = model.decode(input: decoderInput, memory: memory).lastLayerOutput
            let prob = model.generate(input: out[0...,-1])
            let nextWord = Int32(prob.argmax().scalarized())
            ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1)
        }
        return ys
    }
}
