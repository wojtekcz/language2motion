import XCTest
import Foundation
import TensorFlow
import Datasets
import ModelSupport


class CategoricalDistributionHeadTests: XCTestCase {

    func testCatDistHead() throws {
        // + configure forward pass
        // TODO: create new head class
        // TODO: use categorical cross-entropy loss
        // TODO: integrate cce loss with bernoulli loss
        
        print("\n===> setup test")
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        // let device = Device.defaultXLA
        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .micro, device: device)

        var model = ModelFactory.getModel(vocabSize: dsMgr.textProcessor!.vocabulary.count)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")
        model.move(to: device)
//        let _ = model(source)

        let input = source
        
        let encoded = model.encode(input: input.sentence)
        LazyTensorBarrier()
        
        let decoded = model.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
        LazyTensorBarrier()
        
        Context.local.learningPhase = .inference
        let mixtureModelInput = model.preMixtureDense(decoded.lastLayerOutput)
        
        let predsN = model.mixtureModel(mixtureModelInput)
        LazyTensorBarrier()
        
        func roundT(_ num: Tensor<Float>, prec: Int = 4) -> Tensor<Float> {
            return round(num*1e4)/1e4
        }
        
        print("predsN")
        predsN.printPreds()
        print(roundT(predsN.mixtureMeans[0, 0...1]))
        print("===> end test\n")
    }

}
