import Foundation
import ModelSupport
import TensorFlow
import PythonKit


public struct Lang2Lang <Entropy: RandomNumberGenerator> {
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

    public typealias Samples = LazyMapSequence<[Example], TextBatch>
    public let datasetURL: URL
    /// The training texts.
    public let trainingExamples: Samples
    /// The validation texts.
    public let validationExamples: Samples
    // public let validationSamples: [(text: String, label: String)]

    /// The sequence length to which every sentence will be padded.
    public let maxSequenceLength: Int
    public let batchSize: Int
    // public let labels: [String]

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, TextBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, TextBatch>    
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
    ///   - exampleMap: a transform that processes `Example` in `TextBatch`.
    public init(
        datasetURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        entropy: Entropy,
        exampleMap: @escaping (Example) -> TextBatch
    ) throws {
        // Load the data file.
        self.datasetURL = datasetURL
        let df = pd.read_csv(datasetURL.path)
        // labels = df.label.unique().sorted().map {String($0)!}
        let (train_df, test_df) = model_selection.train_test_split(df, test_size: 0.2).tuple2
        
        // validationSamples = Python.list(test_df.iterrows()).map { (text: String($0[1].text)!, label: String($0[1].label)!) }
        trainExamples = Lang2Lang.Df2Example(df: train_df)
        valExamples = Lang2Lang.Df2Example(df: test_df)
        trainingExamples = trainExamples.lazy.map(exampleMap)
        validationExamples = valExamples.lazy.map(exampleMap)
      
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        // Create the training sequence of epochs.
        trainingEpochs = TrainingEpochs(
        samples: trainingExamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, TextBatch> in
            batches.lazy.map{ 
            $0.paddedAndCollated(to: maxSequenceLength)
            }
        }
        
        // Create the validation collection of batches.
        validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
            $0.paddedAndCollated(to: maxSequenceLength)
        }
    }
}

extension Lang2Lang where Entropy == SystemRandomNumberGenerator {
    /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
    /// by `maximumSequenceLength`.
    ///
    /// - Parameter exampleMap: a transform that processes `Example` in `TextBatch`.
    public init(
        datasetURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        exampleMap: @escaping (Example) -> TextBatch
    ) throws {
        try self.init(
        datasetURL: datasetURL,
        maxSequenceLength: maxSequenceLength,
        batchSize: batchSize,
        entropy: SystemRandomNumberGenerator(),
        exampleMap: exampleMap
        )
    }
}
