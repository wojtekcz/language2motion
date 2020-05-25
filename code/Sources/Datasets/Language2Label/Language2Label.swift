import Foundation
import ModelSupport
import TensorFlow
import PythonKit


/// A `TextBatch` with the corresponding labels.
public typealias LabeledTextBatch = (data: TextBatch, label: Tensor<Int32>)


/// Language2Label example.
public struct Language2LabelExample {
    public typealias LabelTuple = (idx: Int, label: String)

    public let id: String
    public let text: String
    public let label: LabelTuple?

    public init(id: String, text: String, label: LabelTuple?) {
        self.id = id
        self.text = text
        self.label = label
    }
}


public struct Language2Label <Entropy: RandomNumberGenerator> {
    public typealias Samples = LazyMapSequence<[Language2LabelExample], LabeledTextBatch>
    public let datasetURL: URL
    /// The training texts.
    public let trainingExamples: Samples
    /// The validation texts.
    public let validationExamples: Samples

    /// The sequence length to which every sentence will be padded.
    public let maxSequenceLength: Int
    public let batchSize: Int
    public let labels: [String]

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, LabeledTextBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, LabeledTextBatch>    
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension Language2Label {

    static func Df2Example(df: PythonObject, labels: [String]) -> [Language2LabelExample] {
        return Python.list(df.iterrows()).map {
            (rowObj: PythonObject) -> Language2LabelExample in 
            let row = rowObj.tuple2.1
            let sample_id: String = "\(row.sample_id)" // Int to String
            let text: String = String(row.text)!
            let labelStr: String? = String(row.label)
            let label: Language2LabelExample.LabelTuple? = Language2LabelExample.LabelTuple(idx: labels.firstIndex(of: labelStr!)!, label: labelStr!)
            return Language2LabelExample(id: sample_id, text: text, label: label)
        }
    }
}

extension Language2Label {
  /// Creates an instance of `datasetURL` dataset with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering. It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is determinstic and not dependent on
  ///     other operations.
  ///   - exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    datasetURL: URL,
    maxSequenceLength: Int,
    batchSize: Int,
    entropy: Entropy,
    exampleMap: @escaping (Language2LabelExample) -> LabeledTextBatch
  ) throws {
    // Load the data file.
        self.datasetURL = datasetURL
        let df = pd.read_csv(datasetURL.path)
        labels = df.label.unique().sorted().map {String($0)!}
        let (train_df, test_df) = model_selection.train_test_split(df, test_size: 0.2).tuple2
        
        trainingExamples = Language2Label.Df2Example(df: train_df, labels: labels).lazy.map(exampleMap)
        validationExamples = Language2Label.Df2Example(df: test_df, labels: labels).lazy.map(exampleMap)
      
      
    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create the training sequence of epochs.
    trainingEpochs = TrainingEpochs(
      samples: trainingExamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
    ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledTextBatch> in
      batches.lazy.map{ 
        (
          data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
          label: Tensor($0.map(\.label))
        )
      }
    }
    
    // Create the validation collection of batches.
    validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength).lazy.map{ 
      (
        data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
        label: Tensor($0.map(\.label))
      )
    }
  }
}

extension Language2Label where Entropy == SystemRandomNumberGenerator {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameter exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    datasetURL: URL,
    maxSequenceLength: Int,
    batchSize: Int,
    exampleMap: @escaping (Language2LabelExample) -> LabeledTextBatch
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
