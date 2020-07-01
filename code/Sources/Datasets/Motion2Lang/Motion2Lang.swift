import Foundation
import ModelSupport
import TensorFlow
import PythonKit


public struct Motion2Lang {
    /// Motion2Lang example.
    public struct Example {
        public let id: String
        // public let sourceSentence: String
        public let motionSample: MotionSample
        public let targetSentence: String

        public init(
            id: String, 
            // sourceSentence: String, 
            motionSample: MotionSample,
            targetSentence: String
        ) {
            self.id = id
            // self.sourceSentence = sourceSentence
            self.motionSample = motionSample
            self.targetSentence = targetSentence
        }
    }

    public let motionDataset: MotionDataset

    public let trainExamples: [Example]
    public let valExamples: [Example]

    public typealias Samples = LazyMapSequence<[Example], MotionLangBatch>

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
        LazyMapSequence<Batches, MotionLangBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, MotionLangBatch>    
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension Motion2Lang {

    // static func Df2Example(df: PythonObject) -> [Example] {
    //     return Python.list(df.iterrows()).map {
    //         (rowObj: PythonObject) -> Example in 
    //         let row = rowObj.tuple2.1
    //         let sample_id: String = "\(row.sample_id)" // Int to String
    //         let text: String = String(row.text)!
    //         // let label: String = String(row.label)!
    //         // return Example(id: sample_id, sourceSentence: text, targetSentence: text)
    //         return Example(id: sample_id, motionSample: text, targetSentence: text)
    //     }
    // }
    public struct LangRec {
        let sampleID: Int
        let text: String
        let label: String
    }
    static func transformDF(df: PythonObject) -> [LangRec] {
        return Python.list(df.iterrows()).map {
            (rowObj: PythonObject) -> LangRec in 
            let row = rowObj.tuple2.1
            let sample_id: Int = Int(row.sample_id)!
            let text: String = String(row.text)!
            let label: String = String(row.label)!
            // return Example(id: sample_id, sourceSentence: text, targetSentence: text)
            return LangRec(sampleID: sample_id, text: text, label: label)
        }
    }

    static func getExample(langRec: LangRec, motionSample: MotionSample) -> Example {
        let sample_id: String = "\(langRec.sampleID)" // Int to String
        return Example(id: sample_id, motionSample: motionSample, targetSentence: langRec.text)
    }
}

extension Motion2Lang {
    /// Creates an instance of `motionDatasetURL` motion dataset with `langDatasetURL` labels,
    /// with batches of size `batchSize` by `maximumSequenceLength`.
    ///
    /// - Parameters:
    ///   - exampleMap: a transform that processes `Example` in `MotionLangBatch`.
    public init(
        motionDatasetURL: URL,
        langDatasetURL: URL,
        maxSequenceLength: Int, // TODO: separate motion from text sequence length?
        batchSize: Int,
        exampleMap: @escaping (Example) -> MotionLangBatch
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset(from: motionDatasetURL)
        print(motionDataset.description)
        let df = pd.read_csv(langDatasetURL.path)
        let (train_df, test_df) = model_selection.train_test_split(df, test_size: 0.2).tuple2
        
        // create LangRecs
        let trainLangRecs = Motion2Lang.transformDF(df: train_df)
        let valLangRecs = Motion2Lang.transformDF(df: test_df)

        // TODO: create Examples
        var _motionsDict: [Int: MotionSample] = [:]
        for ms in motionDataset.motionSamples {
            _motionsDict[ms.sampleID] = ms
        }

        trainExamples = trainLangRecs.map {
            Motion2Lang.getExample(langRec: $0, motionSample: _motionsDict[$0.sampleID]!)
        }
        valExamples = valLangRecs.map {
            Motion2Lang.getExample(langRec: $0, motionSample: _motionsDict[$0.sampleID]!)
        }

        trainingSamples = trainExamples.lazy.map(exampleMap)
        validationSamples = valExamples.lazy.map(exampleMap)
      
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainingEpochs = TrainingEpochs(
        samples: trainingSamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, MotionLangBatch> in
            batches.lazy.map{ 
            // $0.paddedAndCollated(to: maxSequenceLength)
                Motion2Lang.reduceDataBatches(Array($0))
            }
        }
        
        // Create the validation collection of batches.
        validationBatches = validationSamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
            //$0.paddedAndCollated(to: maxSequenceLength)
            Motion2Lang.reduceDataBatches(Array($0))
        }
    }

    static func reduceDataBatches(_ batches: [MotionLangBatch]) -> MotionLangBatch {
        return MotionLangBatch(tokenIds: Tensor(batches.map{ $0.tokenIds.squeezingShape(at: 0) }), // this should be fine
                        targetTokenIds: Tensor(batches.map{ $0.targetTokenIds.squeezingShape(at: 0) }),
                        mask: Tensor(batches.map{ $0.mask.squeezingShape(at: 0) }),
                        targetMask: Tensor(batches.map{ $0.targetMask.squeezingShape(at: 0) }),
                        targetTruth: Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) }),
                        tokenCount: batches.map { $0.tokenCount }.reduce(0, +))
    }

}
