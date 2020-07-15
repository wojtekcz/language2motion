import Foundation
import ModelSupport
import TensorFlow
import PythonKit

public struct Lang2Motion {

    public struct LangRec {
        public let sampleID: Int
        public let text: String
        public let label: String
    }

    public struct Example {
        public let id: String
        public let sentence: String
        public let motionSample: MotionSample

        public init(id: String, sentence: String, motionSample: MotionSample) {
            self.id = id
            self.sentence = sentence
            self.motionSample = motionSample
        }
    }

        public let motionDataset: MotionDataset

    public let langRecs: [LangRec]
    public let langRecsDict: [Int: LangRec]

    public let motionSampleDict: [Int: MotionSample]

    public let trainExamples: [Example]
    public let valExamples: [Example]

    public typealias Samples = LazyMapSequence<[Example], LangMotionBatch>

    /// The training texts.
    public let trainingSamples: Samples
    /// The validation texts.
    public let validationSamples: Samples
    // public let validationSamples: [(text: String, label: String)]

    /// The sequence length to which every sentence will be padded.
    public let maxSequenceLength: Int
    public let batchSize: Int

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, SystemRandomNumberGenerator>, 
        LazyMapSequence<Batches, LangMotionBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, LangMotionBatch>
}

extension Lang2Motion {
    static func transformDF(df: PythonObject) -> [LangRec] {
        return Python.list(df.iterrows()).map {
            (rowObj: PythonObject) -> LangRec in 
            let row = rowObj.tuple2.1
            let sample_id: Int = Int(row.sample_id)!
            let text: String = String(row.text)!
            let label: String = String(row.label)!
            return LangRec(sampleID: sample_id, text: text, label: label)
        }
    }

    public static func getExample(motionSample: MotionSample, langRec: LangRec) -> Example {
        let sample_id: String = "\(langRec.sampleID)" // Int to String
        return Example(id: sample_id, sentence: langRec.text, motionSample: motionSample)
    }
}

extension Lang2Motion {

    public init(
        motionDatasetURL: URL,
        langDatasetURL: URL,
        maxSequenceLength: Int, // TODO: separate motion length from text sequence length?
        batchSize: Int,
        minMotionLength: Int = 10,
        trainTestSplit: Double = 0.8,
        exampleMap: @escaping (Example) -> LangMotionBatch
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset(from: motionDatasetURL)
        print(motionDataset.description)
        let df = pd.read_csv(langDatasetURL.path)

        // filter out samples without annotations
        var motionSamples = motionDataset.motionSamples.filter { $0.annotations.count > 0 }
        print("keeping \(motionSamples.count) annotatated motions")

        // filter out shortest samples
        motionSamples = motionSamples.filter { $0.motionFramesArray.shape[0] >= minMotionLength }
        print("keeping \(motionSamples.count) longer motions, with minimum \(minMotionLength) frames")

        // split into train/test sets
        var trainMotionSamples: [MotionSample] = []
        var testMotionSamples: [MotionSample] = []
        (trainMotionSamples, testMotionSamples) = motionSamples.trainTestSplitMotionSamples(split: trainTestSplit)

        // create LangRecs
        let _langRecs = Lang2Motion.transformDF(df: df)

        // [sampleID:LangRec] mapping
        var _langRecsDict: [Int: LangRec] = [:]
        for langRec in _langRecs {
            _langRecsDict[langRec.sampleID] = langRec
        }

        langRecs = _langRecs
        langRecsDict = _langRecsDict

        // [sampleID:MotionSample] mapping
        var _motionSampleDict: [Int: MotionSample] = [:]
        for ms in motionDataset.motionSamples {
            // only assign first (downsampled) sample
            if _motionSampleDict[ms.sampleID] == nil {
                _motionSampleDict[ms.sampleID] = ms
            }
        }
        motionSampleDict = _motionSampleDict

        // create Examples
        trainExamples = trainMotionSamples.map {
            Lang2Motion.getExample(motionSample: $0, langRec: _langRecsDict[$0.sampleID]!)
        }
        valExamples = testMotionSamples.map {
            Lang2Motion.getExample(motionSample: $0, langRec: _langRecsDict[$0.sampleID]!)
        }

        trainingSamples = trainExamples.lazy.map(exampleMap)
        validationSamples = valExamples.lazy.map(exampleMap)

        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainingEpochs = TrainingEpochs(
        samples: trainingSamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LangMotionBatch> in
            batches.lazy.map{ 
                Lang2Motion.reduceDataBatches(Array($0))
            }
        }
        
        // Create the validation collection of batches.
        validationBatches = validationSamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
            Lang2Motion.reduceDataBatches(Array($0))
        }
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        // var maxLength: Int? = 50 // FIXME: move this out
        // maxLength = maxLength ?? batches.map { $0.motionFrames.shape[1] }.max()!

        // let motionFrames: Tensor<Float> = Tensor(batches.map{$0.motionFrames.paddedOrCropped(to: maxLength!)})

        // // let mask: Tensor<Float> = Tensor(batches.map{$0.mask.paddedOrCropped(to: maxLength!)})        
        // // getting mask from motionFrames, so it's
        // let mask: Tensor<Float> = motionFrames[0...,0...,MotionFrame.cjpMotionFlagIdx].expandingShape(at: 1)

        // // let mask: Tensor<Float> = Tensor(batches.map{ $0.mask.squeezingShape(at: 0) })
        // let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{$0.origMotionFramesCount})

        // let targetTokenIds: Tensor<Int32> = Tensor(batches.map{ $0.targetTokenIds.squeezingShape(at: 0) })
        // let targetMask: Tensor<Float> = Tensor(batches.map{ $0.targetMask.squeezingShape(at: 0) })
        // let targetTruth: Tensor<Int32> = Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) })
        return LangMotionBatch()
        // motionFrames: motionFrames, 
        //                 mask: mask,
        //                 origMotionFramesCount: origMotionFramesCount,
        //                 targetTokenIds: targetTokenIds,
        //                 targetMask: targetMask,
        //                 targetTruth: targetTruth)
    }
}
