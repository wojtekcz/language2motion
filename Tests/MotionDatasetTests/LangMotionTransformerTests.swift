import XCTest
import TensorFlow

import TextModels
import TranslationModels
import ModelSupport
import Datasets

import LangMotionModels

class LangMotionTransformerTests: XCTestCase {

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    var dataset: Lang2Motion? = nil
    var textProcessor: TextProcessor? = nil
    let maxTextSequenceLength =  40
    let maxMotionLength =  150
    let batchSize = 10

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func loadDataset(datasetSize: DatasetSize, device: Device) throws -> Lang2Motion {
        /// load dataset
        print("\nLoading dataset...")

        let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")
        var discretizer = MotionDiscretizer(n_bins: 300)

        let dataset = try Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: batchSize,
            minMotionLength: 20,
            maxMotionLength: 150,
            discretizer: &discretizer,
            trainTestSplit: 1.0,
            device: device
        ) { [self] (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = self.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: self.maxTextSequenceLength)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer)

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source, label: target)
            return singleBatch
        }
        print("Dataset acquired.")
        return dataset
    }
    
    func getTextProcessor() -> TextProcessor {
        /// instantiate text processor
        print("instantiate text processor")
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        return TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
    }
    
    func getModel(vocabSize: Int) -> LangMotionTransformer {
        let config = LangMotionTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            nbMixtures: 20,
            layerCount: 6,
            encoderDepth: 256,
            decoderDepth: 512,
            feedForwardSize: 2048,
            headCount: 16,
            dropoutProbability:  0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            mixtureDepth: 1500,
            activation: relu
        )

        return LangMotionTransformer(config: config)
    }
    
    func testTimeDistributedMixtureModel() throws {
        print("\n===> setup test")
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        // let device = Device.defaultXLA
        let device = Device.defaultTFEager
        print("backend: \(device)")

        textProcessor = getTextProcessor()
        var discretizer = MotionDiscretizer(n_bins: 300)
        dataset = try! loadDataset(datasetSize: .micro, device: device)
        var model = getModel(vocabSize: textProcessor!.vocabulary.count)
        
        let motionSample = dataset!.motionSamples[0]
        let sentence = textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)

        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")
        model.move(to: device)
        // let _ = model(source)
        let input = source
        time {
            print(1)
            let encoded = model.encode(input: input.sentence)
            LazyTensorBarrier()

            print(2)
            let decoded = model.decode(sourceMask: input.sourceAttentionMask, motionPart: input.motionPart, memory: encoded.lastLayerOutput)
            LazyTensorBarrier()

            time {
                print(3)
                Context.local.learningPhase = .inference
                let mixtureModelInput = decoded.lastLayerOutput
//                let predsO = model.mixtureModel.callAsFunction_old(mixtureModelInput)
                let predsO = model.mixtureModel(mixtureModelInput)
                let predsN = model.mixtureModel(mixtureModelInput)
                LazyTensorBarrier()

                // func roundT(_ num: Tensor<Float>, prec: Int = 4) -> Tensor<Float> {
                //     return round(num*1e4)/1e4
                // }

                // TODO: compare mixture model old and new outputs
                print("mixtureMeans ", (predsO.mixtureMeans - predsN.mixtureMeans).sum().scalar! < 1e-3)
                print("mixtureVars:", (predsO.mixtureVars - predsN.mixtureVars).sum().scalar! < 1e-3)
                print("mixtureWeights:", (predsO.mixtureWeights - predsN.mixtureWeights).sum().scalar! < 1e-3)
                print("stops:", predsO.stops == predsN.stops)

                // predsO.printPreds()
                // print("predsO")
                // print(roundT(predsO.mixtureMeans[0, 0...1]))
                // print("predsN")
                // // predsN.printPreds()
                // print(roundT(predsN.mixtureMeans[0, 0...1]))
            }
            print(4)
            // let rslt = LangMotionTransformerOutput(preds: preds, encoded: encoded, decoded: decoded)
        }
        print("===> end test\n")
    }

    func testForwardPass() throws {
        print("\n===> setup test")
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        let device = Device.defaultXLA
        // let device = Device.defaultTFEager
        print("backend: \(device)")

        textProcessor = getTextProcessor()
        var discretizer = MotionDiscretizer(n_bins: 300)
        dataset = try! loadDataset(datasetSize: .micro, device: device)
        var model = getModel(vocabSize: textProcessor!.vocabulary.count)
        
        let motionSample = dataset!.motionSamples[0]
        let sentence = textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer)

        var source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)

        source = LangMotionBatch.Source(copying: source, to: device)

        print("\n===> start test")
        model.move(to: device)
        time {
            let _ = model(source)
        }
        print("===> end test\n")
    }

    func testX10Performance() throws {
        let _ = _ExecutionContext.global

        /// Select eager or X10 backend
        // let device = Device.defaultXLA
        let device = Device.defaultTFEager
        print("backend: \(device)")

        textProcessor = getTextProcessor()
        var discretizer = MotionDiscretizer(n_bins: 300)
        dataset = try! loadDataset(datasetSize: .micro, device: device)
        let model = getModel(vocabSize: textProcessor!.vocabulary.count)
        
        let motionSample = dataset!.motionSamples[0]
        let sentence = textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
        let (motionPart, _) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer)

        let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)

        self.measure {
            let _ = model(source)
            LazyTensorBarrier()
        }
    }

   static var allTests = [
       ("testForwardPass", testForwardPass),
       ("testX10Performance", testX10Performance)
   ]
}
