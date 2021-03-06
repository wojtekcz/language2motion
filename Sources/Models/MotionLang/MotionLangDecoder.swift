import Foundation
import TensorFlow
import Datasets

extension Tensor {
    public func copy(to device: Device) -> Tensor<Scalar> {
        return Tensor<Scalar>(copying: self, to: device)
    }
}

public class MotionLangDecoder {
    public static func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch.MLSource, maxLength: Int, startSymbol: Int32, device: Device) -> Tensor<Int32> {
        let memory = model.encode(input: input).lastLayerOutput
        // TODO: make loop work on tensors of same size, so it can be compiled with X10 backend
        var ys = Tensor(repeating: startSymbol, shape: [1,1], on: device)
        for _ in 0..<maxLength {
            let targetFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]], on: device)
            let targetMask = MotionLangBatch.makeSelfAttentionDecoderMask(target: targetFlag, pad: 0, on: device)

            let decoderInput = MotionLangBatch.MLSource(sampleID: input.sampleID, motion: input.motion,
                                         mask: input.mask,
                                         origMotionFramesCount: input.origMotionFramesCount,
                                         targetTokenIds: ys,
                                         targetMask: targetMask
                                         )
            let decoded = model.decode(input: decoderInput, memory: memory).lastLayerOutput
            let prob = model.generate(input: decoded[0...,-1])
            let nextWord = Int32(prob.argmax().scalarized())
            ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1]).copy(to: device)], alongAxis: 1)
        }
        return ys
    }
}
