import TensorFlow
import Datasets


public protocol MotionClassifierProtocol: Differentiable {
    @noDerivative
    var maxSequenceLength: Int {get}

    /// Returns: logits with shape `[batchSize, classCount]`.
    @differentiable(wrt: self)
    func callAsFunction(_ input: MotionBatch) -> Tensor<Float>
}

public struct Prediction {
    public let classIdx: Int
    public let className: String
    public let probability: Float
}

extension MotionClassifierProtocol {
    public func predict(motionSamples: [MotionSample], labels: [String], batchSize: Int, device: Device) -> [Prediction] {
        Context.local.learningPhase = .inference
        let validationExamples = motionSamples.map {
            (example) -> MotionBatch in
            let motionFrames = Tensor<Float>(example.motionFramesArray)
            let mfIdx = MotionFrame.cjpMotionFlagIdx
            let motionFlag = Tensor<Int32>(motionFrames[0..., mfIdx...mfIdx].squeezingShape(at: 1))
            let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))
            let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag, origMotionFramesCount: origMotionFramesCount)
            return motionBatch
        }

        let validationBatches = validationExamples.inBatches(of: batchSize).map { 
            $0.paddedAndCollated(to: maxSequenceLength)
        }

        var preds: [Prediction] = []
        for eagerBatch in validationBatches {
            let batch = MotionBatch(copying: eagerBatch, to: device)
            let logits = self(batch)
            let probs = softmax(logits, alongAxis: 1)
            let classIdxs = logits.argmax(squeezingAxis: 1)
            LazyTensorBarrier()
            let batchPreds = (0..<classIdxs.shape[0]).map { 
                (idx) -> Prediction in
                let classIdx: Int = Int(classIdxs[idx].scalar!)
                let prob = probs[idx, classIdx].scalar!
                return Prediction(classIdx: classIdx, className: labels[classIdx], probability: prob)
            }
            preds.append(contentsOf: batchPreds)
        }
        return preds
    }

}
