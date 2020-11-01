import XCTest
import Foundation
import TensorFlow
import Datasets
import ModelSupport
import LangMotionModels
import PythonKit


func roundT(_ num: Tensor<Float>, prec: Int = 4) -> Tensor<Float> {
    return round(num*1e4)/1e4
}


class CategoricalDistributionHeadTests: XCTestCase {

    func testCatDistHead() throws {
        // + configure forward pass
        // + create new head class
        // + sample & argmax
        // TODO: try to make sampling faster with a tensorflow call
        // + de-discretize
        // + de-scale
        // + use categorical cross-entropy loss
        // TODO: integrate cce loss with bernoulli loss
        
        print("\n===> setup test")
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        // let device = Device.defaultXLA
        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .micro, device: device)

        var model = ModelFactory.getModel(vocabSize: dsMgr.textProcessor!.vocabulary.count)
        let catDistHead = MotionCatDistHead(inputSize: model.config.decoderDepth, nbJoints: model.config.nbJoints, discreteBins: 300)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")
        model.move(to: device)

        let input = source
        
        let encoded = model.encode(input: input.sentence)
        LazyTensorBarrier()
        let decoded = model.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        LazyTensorBarrier()
        
        Context.local.learningPhase = .inference
        //let mixtureModelInput = model.preMixtureDense(decoded.lastLayerOutput)
        let mixtureModelInput = decoded.lastLayerOutput

        print("input")
        print("input.shape: \(mixtureModelInput.shape)")
        print(roundT(mixtureModelInput))
        
        let preds: MotionCatDistPreds = catDistHead(mixtureModelInput)
        LazyTensorBarrier()
        
        preds.printPreds()
        print("catDistProbs")
        print(roundT(preds.catDistProbs))
        let sums = preds.catDistProbs.sum(alongAxes: 3)
        print("sums.shape: \(sums.shape)")
        print(roundT(sums))
        
        // + use categorical cross-entropy loss
        let labels = target.discreteMotion.reshaped(to: [-1])
        let sh = target.discreteMotion.shape
        let resultSize =  sh[0] * sh[1] * sh[2]
        let logits = preds.catDistProbs.reshaped(to: [resultSize, -1])
        
        let loss = softmaxCrossEntropy(logits: logits, labels: labels, reduction: _mean)
        print("loss: \(loss)")

        
        // + sample & argmax & de-discretize & de-scale
        let np = Python.import("numpy")
        func sampleCatDistMotion(catDistProbs: Tensor<Float>) -> Tensor<Int32> {
            var samples: [Int32] = []
            let s = catDistProbs.shape
            let (bs, nFrames, nbJoints) = (s[0], s[1], s[2])
            for s in 0..<bs {
                for t in 0..<nFrames {
                    for j in 0..<nbJoints {
                        let pvals = catDistProbs[s, t, j].scalars.map { Double($0)}
                        // TODO: try to make sampling faster with a tensorflow call
                        let sample: Int32 = Int32(np.argmax(np.random.multinomial(1, pvals)))!
                        samples.append(sample)
                    }
                }
            }
            let samplesTensor = Tensor<Int32>(shape: [bs, nFrames, nbJoints], scalars: samples)
            return samplesTensor
        }
        
        let samplesTensor = sampleCatDistMotion(catDistProbs: preds.catDistProbs)
        print("samplesTensor.shape: \(samplesTensor.shape)")
        print("samplesTensor: \(samplesTensor)")

        // de-discretize
        let invSamplesTensor = dsMgr.discretizer!.inverse_transform(samplesTensor)
        print("invSamplesTensor.shape: \(invSamplesTensor.shape)")
        print("invSamplesTensor: \(roundT(invSamplesTensor))")
        // de-scale
        let descaledSamplesTensor = dsMgr.dataset!.scaler.inverse_transform(invSamplesTensor)
        print("descaledSamplesTensor.shape: \(descaledSamplesTensor.shape)")
        print("descaledSamplesTensor: \(roundT(descaledSamplesTensor))")

        print("===> end test\n")
    }

}
