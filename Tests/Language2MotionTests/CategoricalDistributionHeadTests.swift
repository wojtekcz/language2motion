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

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    
    func testCatDistHead() throws {
        // + configure forward pass
        // + create new head class
        // + sample & argmax
        // TODO: try to make sampling faster with a tensorflow call
        // + de-discretize
        // + de-scale
        // + use categorical cross-entropy loss
        // + integrate cce loss with bernoulli loss
        
        print("\n===> setup test")
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        // let device = Device.defaultXLA
        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: device)

        let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)

        let model = ModelFactory.getModel4(vocabSize: dsMgr.textProcessor!.vocabulary.count, logdirURL: logdirURL)

        let catDistHead = MotionCatDistHead(inputSize: model.config.decoderDepth, nbJoints: model.config.nbJoints, discreteBins: 300)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")
//        model.move(to: device)

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
        //print("catDistProbs")
        //print(roundT(preds.catDistProbs))
        //let sums = preds.catDistProbs.sum(alongAxes: 3)
        //print("sums.shape: \(sums.shape)")
        //print(roundT(sums))
        
        // + use categorical cross-entropy loss
        let labels = target.discreteMotion.reshaped(to: [-1])
        let sh = target.discreteMotion.shape
        let resultSize =  sh[0] * sh[1] * sh[2]
        let logits = preds.catDistLogits.reshaped(to: [resultSize, -1])
        
        @differentiable
        func _none(t: Tensor<Float>) -> Tensor<Float> { t }
        let sceLoss = softmaxCrossEntropy(logits: logits, labels: labels, reduction: _none)
        print("sceLoss.shape: \(sceLoss.shape)")
        print("sceLoss: \(sceLoss)")
        
        // + integrate sce loss with bernoulli loss
        let args = CDLossArgs(
            device: device
        )
        let cdsLoss = categoryDistributionSurrogateLoss(y_pred: preds, y_true: target, args: args)
        print("cdsLoss: \(cdsLoss)")

        print("===> end test\n")
    }

    func testCatDistHeadMotionGeneration() throws {
        print("\n===> setup test")

        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: device)

        let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)

        let model = ModelFactory.getModel4(vocabSize: dsMgr.textProcessor!.vocabulary.count, logdirURL: logdirURL)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")

        let input = source
        
        
        Context.local.learningPhase = .inference
        
        let preds = model(input)
        
        print()
        // + sample & argmax & de-discretize & de-scale
        
        let samplesTensor = MotionCatDistDecoder.sampleCatDistMotion(catDistLogits: preds.preds.catDistLogits)
        print("samplesTensor.shape: \(samplesTensor.shape)")
        print("samplesTensor: \(samplesTensor)")

        // de-discretize
        let invSamplesTensor = dsMgr.discretizer!.inverse_transform(samplesTensor)
        print("invSamplesTensor.shape: \(invSamplesTensor.shape)")
        print("invSamplesTensor: \(roundT(invSamplesTensor))")
        // de-scale
        let descaledSamplesTensor = invSamplesTensor//dsMgr.dataset!.scaler.inverse_transform(invSamplesTensor)
        print("descaledSamplesTensor.shape: \(descaledSamplesTensor.shape)")
        print("descaledSamplesTensor: \(roundT(descaledSamplesTensor))")

        print("===> end test\n")
    }

    func testCatDistHeadMotionDecoding() throws {
        print("\n===> setup test")

        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: device)

        let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)

        let model = ModelFactory.getModel4(vocabSize: dsMgr.textProcessor!.vocabulary.count, logdirURL: logdirURL)
        let motionDecoder = MotionCatDistDecoder(discretizer: dsMgr.discretizer!, transformer: model)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)

        print("\n===> start test")

        Context.local.learningPhase = .inference
        let (decodedMotion, _) = motionDecoder.greedyDecodeMotion(sentence: sentence, startMotion: nil, maxMotionLength: 50)
        
        // de-scale
        let descaledMotion = decodedMotion //dsMgr.dataset!.scaler.inverse_transform(decodedMotion)
        print("descaledMotion.shape: \(descaledMotion.shape)")
        print("descaledMotion: \(roundT(descaledMotion))")
        
        print("===> end test\n")
    }

    func testMotionDescalingIssue() throws {
        print("\n===> setup test")

        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: device)

        let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)

//        let model = ModelFactory.getModel6(vocabSize: dsMgr.textProcessor!.vocabulary.count, logdirURL: logdirURL)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")

//        let input = source
        
        
        Context.local.learningPhase = .inference
        
//        let preds = model(input)
        
        // loss?
//        let args = CDLossArgs(
//            device: device
//        )
//        let cdsLoss = categoryDistributionSurrogateLoss(y_pred: preds.preds, y_true: target, args: args)
//        print("cdsLoss: \(cdsLoss)")
        
        // show input sample
        let sampleMotion = target.discreteMotion
        print("sampleMotion.shape: \(sampleMotion.shape)")
        print("sampleMotion: \(sampleMotion)")

        // decode motion
        let decodedMotion = sampleMotion
//        let decodedMotion = MotionCatDistDecoder.sampleCatDistMotion(catDistLogits: preds.preds.catDistLogits)
//        print("decodedMotion.shape: \(decodedMotion.shape)")
//        print("decodedMotion: \(decodedMotion)")

//        let stops = preds.preds.stops
//        print("stops.shape: \(stops.shape)")
//        print("stops: \(stops.squeezingShape(at: [0, 2]))")

        print()
        // show original motion
//        let sourceMotion = source.motionPart.motion[0]
//        print("sourceMotion.shape: \(sourceMotion.shape)")
//        print("sourceMotion: \(roundT(sourceMotion))")

//        let targetMotion = target.motion[0]
//        print("targetMotion.shape: \(targetMotion.shape)")
//        print("targetMotion: \(roundT(targetMotion))")

        // de-discretize
        let dediscretizedMotion = dsMgr.discretizer!.inverse_transform(decodedMotion)[0]
        print("dediscretizedMotion.shape: \(dediscretizedMotion.shape)")
        print("dediscretizedMotion: \(roundT(dediscretizedMotion))")

        // de-scale
        let descaledMotion = dsMgr.dataset!.scaler.inverse_transform(dediscretizedMotion)
        print("descaledMotion.shape: \(descaledMotion.shape)")
        print("descaledMotion: \(roundT(descaledMotion))")

        // save mmm file(s)
        func saveMotionToMMM(dataset: Lang2Motion, motion: Tensor<Float>, mmmURL: URL) {
            let jointNames = dataset.motionSamples[0].jointNames
            let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: motion)
            try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
            print("Saved motion: \(mmmURL.path)")
        }
        
        let rundirURL = logdirURL.appendingPathComponent("run_142")
//        saveMotionToMMM(dataset: dsMgr.dataset!, motion: sourceMotion, mmmURL: rundirURL.appendingPathComponent("sourceMotion.mmm.xml"))
//        saveMotionToMMM(dataset: dsMgr.dataset!, motion: targetMotion, mmmURL: rundirURL.appendingPathComponent("targetMotion.mmm.xml"))
        saveMotionToMMM(dataset: dsMgr.dataset!, motion: dediscretizedMotion, mmmURL: rundirURL.appendingPathComponent("dediscretizedMotion.mmm.xml"))
        saveMotionToMMM(dataset: dsMgr.dataset!, motion: descaledMotion, mmmURL: rundirURL.appendingPathComponent("descaledMotion.mmm.xml"))

//        _Raw.dequantize(<#T##input: Tensor<TensorFlowScalar>##Tensor<TensorFlowScalar>#>, minRange: <#T##Tensor<Float>#>, maxRange: <#T##Tensor<Float>#>)
//        _Raw.fakeQuantWithMinMaxVarsPerChannel(inputs: <#T##Tensor<Float>#>, min: <#T##Tensor<Float>#>, max: <#T##Tensor<Float>#>, numBits: <#T##Int64#>, narrowRange: <#T##Bool#>)
        
        print("===> end test\n")
    }
    
    func testLossFunction() throws {
        print("\n===> setup test")

        let device = Device.defaultTFEager
        print("backend: \(device)")
        
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: device)

        let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)
        let model = ModelFactory.getModel4(vocabSize: dsMgr.textProcessor!.vocabulary.count, logdirURL: logdirURL)
        
        let motionSample = dsMgr.dataset!.motionSamples[0]
        let sentence = dsMgr.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: dsMgr.maxTextSequenceLength)
        let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: dsMgr.maxMotionLength, discretizer: dsMgr.discretizer!)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")

        let input = source
        
        Context.local.learningPhase = .inference
        
        let preds = model(input)
        
        // loss?
        let args = CDLossArgs(
            device: device
        )
        let cdsLoss = categoryDistributionSurrogateLoss(y_pred: preds.preds, y_true: target, args: args)
        print("cdsLoss: \(cdsLoss)")
    }
}
