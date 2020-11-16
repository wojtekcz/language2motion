import XCTest
import TensorFlow

import TextModels
import TranslationModels
import ModelSupport
import Datasets

import LangMotionModels


class LangMotionTransformerTests: XCTestCase {
    
//    func testTimeDistributedMixtureModel() throws {
//        print("\n===> setup test")
//        let _ = _ExecutionContext.global
//
//        /// Select eager or X10 backend
//        // let device = Device.defaultXLA
//        let device = Device.defaultTFEager
//        print("backend: \(device)")
//
//        let dsMgr = DatasetManager(datasetSize: .micro, device: device)
//
//        var model = ModelFactory.getModel(vocabSize: dsMgr.textProcessor!.vocabulary.count)
//
//        let motionSample = dsMgr.dataset!.motionSamples[0]
//        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
//        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)
//
//        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
//
//        source = LangMotionBatch.Source(copying: source, to: device)
//
//        print("\n===> start test")
//        model.move(to: device)
//        // let _ = model(source)
//        let input = source
//        time {
//            print(1)
//            let encoded = model.encode(input: input.sentence)
//            LazyTensorBarrier()
//
//            print(2)
//            let decoded = model.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
//            LazyTensorBarrier()
//
//            time {
//                print(3)
//                Context.local.learningPhase = .inference
//                let mixtureModelInput = decoded.lastLayerOutput
////                let predsO = model.mixtureModel.callAsFunction_old(mixtureModelInput)
//                let predsO = model.mixtureModel(mixtureModelInput)
//                let predsN = model.mixtureModel(mixtureModelInput)
//                LazyTensorBarrier()
//
//                // func roundT(_ num: Tensor<Float>, prec: Int = 4) -> Tensor<Float> {
//                //     return round(num*1e4)/1e4
//                // }
//
//                // TODO: compare mixture model old and new outputs
//                print("mixtureMeans ", (predsO.mixtureMeans - predsN.mixtureMeans).sum().scalar! < 1e-3)
//                print("mixtureVars:", (predsO.mixtureVars - predsN.mixtureVars).sum().scalar! < 1e-3)
//                print("mixtureWeights:", (predsO.mixtureWeights - predsN.mixtureWeights).sum().scalar! < 1e-3)
//                print("stops:", predsO.stops == predsN.stops)
//
//                // predsO.printPreds()
//                // print("predsO")
//                // print(roundT(predsO.mixtureMeans[0, 0...1]))
//                // print("predsN")
//                // // predsN.printPreds()
//                // print(roundT(predsN.mixtureMeans[0, 0...1]))
//            }
//            print(4)
//            // let rslt = LangMotionTransformerOutput(preds: preds, encoded: encoded, decoded: decoded)
//        }
//        print("===> end test\n")
//    }

    func testForwardPass() throws {
//        print("\n===> setup test")
//        let _ = _ExecutionContext.global
//
//        /// Select eager or X10 backend
//        // let device = Device.defaultXLA
//        let device = Device.defaultTFEager
//        print("backend: \(device)")
//
//        let dsMgr = DatasetManager(datasetSize: .micro, device: device)
//
//        var model = ModelFactory.getModel(vocabSize: dsMgr.textProcessor!.vocabulary.count)
//
//        let motionSample = dsMgr.dataset!.motionSamples[0]
//        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
//        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)
//
//        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
//
//        source = LangMotionBatch.Source(copying: source, to: device)
//
//        print("\n===> start test")
//        model.move(to: device)
//        time {
//            let _ = model(source)
//        }
//        print("===> end test\n")
    }

    func testX10Performance() throws {
//        let _ = _ExecutionContext.global
//
//        /// Select eager or X10 backend
//        // let device = Device.defaultXLA
//        let device = Device.defaultTFEager
//        print("backend: \(device)")
//
//        let dsMgr = DatasetManager(datasetSize: .micro, device: device)
//        let model = ModelFactory.getModel(vocabSize: dsMgr.textProcessor!.vocabulary.count)
//
//        let motionSample = dsMgr.dataset!.motionSamples[0]
//        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
//        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)
//
//        let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
//
//        self.measure {
//            let _ = model(source)
//            LazyTensorBarrier()
//        }
    }

   static var allTests = [
       ("testForwardPass", testForwardPass),
       ("testX10Performance", testX10Performance)
   ]
}
