import Foundation
import ModelSupport
import TensorFlow
import PythonKit


public struct Lang2Lang {
    /// Lang2Lang example.
    public struct Example {
        public let id: String
        public let sourceSentence: String
        public let targetSentence: String

        public init(id: String, sourceSentence: String, targetSentence: String) {
            self.id = id
            self.sourceSentence = sourceSentence
            self.targetSentence = targetSentence
        }
    }

    public let trainExamples: [Example]
    public let valExamples: [Example]

    public typealias Samples = LazyMapSequence<[Example], TranslationBatch>
    public let datasetURL: URL
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
        LazyMapSequence<Batches, TranslationBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, TranslationBatch>    
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension Lang2Lang {

    static func Df2Example(df: PythonObject) -> [Example] {
        return Python.list(df.iterrows()).map {
            (rowObj: PythonObject) -> Example in 
            let row = rowObj.tuple2.1
            let sample_id: String = "\(row.sample_id)" // Int to String
            let text: String = String(row.text)!
            return Example(id: sample_id, sourceSentence: text, targetSentence: text)
        }
    }
}

extension Lang2Lang {
    /// Creates an instance of `datasetURL` dataset with batches of size `batchSize`
    /// by `maximumSequenceLength`.
    ///
    /// - Parameters:
    ///   - entropy: a source of randomness used to shuffle sample ordering. It
    ///     will be stored in `self`, so if it is only pseudorandom and has value
    ///     semantics, the sequence of epochs is determinstic and not dependent on
    ///     other operations.
    ///   - exampleMap: a transform that processes `Example` in `TranslationBatch`.
    public init(
        datasetURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        exampleMap: @escaping (Example) -> TranslationBatch
    ) throws {
        // Load the data file.
        self.datasetURL = datasetURL
        let df = pd.read_csv(datasetURL.path)
        let (train_df, test_df) = model_selection.train_test_split(df, test_size: 0.2).tuple2
        
        // validationSamples = Python.list(test_df.iterrows()).map { (text: String($0[1].text)!, label: String($0[1].label)!) }
        trainExamples = Lang2Lang.Df2Example(df: train_df)
        valExamples = Lang2Lang.Df2Example(df: test_df)
        trainingSamples = trainExamples.lazy.map(exampleMap)
        validationSamples = valExamples.lazy.map(exampleMap)
      
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainingEpochs = TrainingEpochs(
        samples: trainingSamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, TranslationBatch> in
            batches.lazy.map{ 
            // $0.paddedAndCollated(to: maxSequenceLength)
                Lang2Lang.reduceDataBatches(Array($0))
            }
        }
        
        // Create the validation collection of batches.
        validationBatches = validationSamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
            //$0.paddedAndCollated(to: maxSequenceLength)
            Lang2Lang.reduceDataBatches(Array($0))
        }
    }

    static func reduceDataBatches(_ batches: [TranslationBatch]) -> TranslationBatch {
        return TranslationBatch(tokenIds: Tensor(batches.map{ $0.tokenIds.squeezingShape(at: 0) }), // this should be fine
                        targetTokenIds: Tensor(batches.map{ $0.targetTokenIds.squeezingShape(at: 0) }),
                        mask: Tensor(batches.map{ $0.mask.squeezingShape(at: 0) }),
                        targetMask: Tensor(batches.map{ $0.targetMask.squeezingShape(at: 0) }),
                        targetTruth: Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) }),
                        tokenCount: batches.map { $0.tokenCount }.reduce(0, +))
    }

}
