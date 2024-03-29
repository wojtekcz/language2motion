import TensorFlow
import PythonKit
import Datasets

let np  = Python.import("numpy")

public func randomNumber(probabilities: [Double]) -> Int {
    // https://stackoverflow.com/questions/30309556/generate-random-numbers-with-a-given-distribution
    // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
    let sum = probabilities.reduce(0, +)
    // Random number in the range 0.0 <= rnd < sum :
    let rnd = Double.random(in: 0.0 ..< sum)
    // Find the first interval of accumulated probabilities into which `rnd` falls:
    var accum = 0.0
    for (i, p) in probabilities.enumerated() {
        accum += p
        if rnd < accum {
            return i
        }
    }
    // This point might be reached due to floating point inaccuracies:
    return (probabilities.count - 1)
}

public class MotionCatDistDecoder {
    
    let discretizer: MotionDiscretizer
    let transformer: LangMotionCatDistTransformer
    
    public init(discretizer: MotionDiscretizer, transformer: LangMotionCatDistTransformer) {
        self.discretizer = discretizer
        self.transformer = transformer
    }
    
    public func greedyDecodeMotion(sentence: LangMotionBatch.Sentence, startMotion: Tensor<Float>?, maxMotionLength: Int) -> (motion: Tensor<Float>, done: Tensor<Int32>)
    {
        // print("\nEncode:")
        // print("======")
        let encoded = transformer.encode(input: sentence)
        let memory = encoded.lastLayerOutput
        // print("  memory.count: \(memory.shape)")

        // print("\nGenerate:")
        // print("=========")

        // TODO: kill ys
        // TODO: gather discretized neutral motion frame
        // start with tensor for neutral motion frame
        let neutralMotionFrame = LangMotionBatch.neutralMotionFrame().expandingShape(at: 0)
        var ys: Tensor<Float> = neutralMotionFrame
        // or with supplied motion
        if startMotion != nil {
            ys = Tensor<Float>(concatenating: [neutralMotionFrame, startMotion!.expandingShape(at:0)], alongAxis: 1)
        }
        // print("ys.shape: \(ys.shape)")
        var discrete_ys = discretizer.transform(ys)
        
        var dones: [Tensor<Int32>] = []

        let maxMotionLength2 = maxMotionLength-ys.shape[1]+1

        for i in 0..<maxMotionLength2 {
            //print("frame: \(f)")
            // print(".", terminator:"")
            // prepare input
            let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
            let window_size = min(i, 5)
            let motionPartMask = LangMotionBatch.makeSelfAttentionDecoderMask(target: motionPartFlag, pad: 0).bandPart(window_size, 0)

            // FIXME: woarkaround, can't assign to tensor int32 element
            var segmentIDs = Tensor<Float>(repeating: Float(LangMotionBatch.MotionSegment.motion.rawValue), shape: [1, ys.shape[1], 1])//.expandingShape(at: 2)
            segmentIDs[0, 0, 0] = Tensor<Float>(Float(LangMotionBatch.MotionSegment.start.rawValue))

            
            let motionPart = LangMotionBatch.MotionPart(discreteMotion: discrete_ys, decSelfAttentionMask: motionPartMask,
                                                        motionFlag: motionPartFlag.expandingShape(at: 2), segmentIDs: Tensor<Int32>(segmentIDs))

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            // decode motion
            let decoded = transformer.decode(sourceMask: source.sourceAttentionMask, motionPart: motionPart, memory: memory)
                        
            // let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
            let catDistInput = decoded.lastLayerOutput
            let catDistInput2 = catDistInput[0...,-1].expandingShape(at: 0)
            let singlePreds = transformer.catDistHead(catDistInput2)
            
            // perform sampling
            let (sampledMotion, _, done) = Self.performNormalMixtureSampling(
                preds: singlePreds, maxMotionLength: maxMotionLength)
            
            //print("sampledMotion: \(sampledMotion)")
            
            // concatenate motion
            discrete_ys = Tensor(concatenating: [discrete_ys, sampledMotion], alongAxis: 1)
            ys = Tensor(concatenating: [ys, discretizer.inverse_transform(sampledMotion)], alongAxis: 1)
            
            // get done signal out
            dones.append(done)
        }
        print()
        let dones2 = Tensor<Int32>(concatenating: dones, alongAxis: 0)
        
        ys = discretizer.inverse_transform(discrete_ys)
        
        return (motion: ys.squeezingShape(at:0)[1...], done: dones2)
    }
    
    public static func performNormalMixtureSampling(preds: MotionCatDistPreds, maxMotionLength: Int) -> (motion: Tensor<Int32>, log_probs: [Float], done: Tensor<Int32>) {
        let motion = Self.sampleCatDistMotion(catDistLogits: preds.catDistLogits)
        let t1 = Tensor<Int32>([[0]])
        return (motion: motion, log_probs: [], done: t1)
    }
    
    //let np = Python.import("numpy")
    public static func sampleCatDistMotion(catDistLogits: Tensor<Float>) -> Tensor<Int32> {
        var samples: [Int32] = []
        let sh = catDistLogits.shape
        let (bs, nFrames, nbJoints) = (sh[0], sh[1], sh[2])
        for s in 0..<bs {
            for f in 0..<nFrames {
                for j in 0..<nbJoints {
                    let pvals = softmax(catDistLogits[s, f, j]).scalars.map { Double($0)}
                    // TODO: try to make sampling faster with a tensorflow call
                    let sample: Int32 = Int32(randomNumber(probabilities: pvals))
                    //let sample: Int32 = Int32(np.argmax(np.random.multinomial(1, pvals)))!
                    samples.append(sample)
                }
            }
        }
        let samplesTensor = Tensor<Int32>(shape: [bs, nFrames, nbJoints], scalars: samples)
        return samplesTensor
    }
}
