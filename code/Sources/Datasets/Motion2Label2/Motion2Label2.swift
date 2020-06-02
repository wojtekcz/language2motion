import Foundation
import ModelSupport
import TensorFlow


/// A `MotionBatch` with the corresponding labels.
public typealias LabeledMotionBatch = (data: MotionBatch, label: Tensor<Int32>)


/// Motion2Label example.
public struct Motion2LabelExample {
    public typealias LabelTuple = (idx: Int, label: String)

    public let id: String
    public let motionSample: MotionSample
    public let label: LabelTuple?

    public init(id: String, motionSample: MotionSample, label: LabelTuple?) {
        self.id = id
        self.motionSample = motionSample
        self.label = label
    }
}


public struct Motion2Label2 <Entropy: RandomNumberGenerator> {
    public typealias Samples = LazyMapSequence<[Motion2LabelExample], LabeledMotionBatch>

    public let motionData: MotionData
    public let trainingExamples: Samples
    public let validationExamples: Samples

    /// The sequence length to which every motion will be padded.
    public let maxSequenceLength: Int
    public let batchSize: Int
    public let labels: [String]
    public let labelsDict: [Int: String]

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, LabeledMotionBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainingEpochs: TrainEpochs
    /// The validation batches.
    public var validationBatches: LazyMapSequence<Slices<Samples>, LabeledMotionBatch>
}


extension Motion2Label2 {

    /// Creates an instance of `serializedDatasetURL` motion dataset with `labelsURL` labels,
    /// with batches of size `batchSize` by `maximumSequenceLength`.
    ///
    /// - Parameters:
    ///   - entropy: a source of randomness used to shuffle sample ordering. It
    ///     will be stored in `self`, so if it is only pseudorandom and has value
    ///     semantics, the sequence of epochs is determinstic and not dependent on
    ///     other operations.
    ///   - exampleMap: a transform that processes `Motion2LabelExample` in `LabeledMotionBatch`.
    public init(
        serializedDatasetURL: URL,
        labelsURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        entropy: Entropy,
        exampleMap: @escaping (Motion2LabelExample) -> LabeledMotionBatch
    ) throws {
        // Load the data files.
        motionData = MotionData(from: serializedDatasetURL)
        print(motionData.description)
        
        let df = pd.read_csv(labelsURL.path)
        let _labels = df.label.unique().sorted().map { String($0)! }

        var _labelsDict: [Int: String] = [:]
        for pythonTuple in df.iterrows() {
            _labelsDict[Int(pythonTuple[1].sample_id)!] = String(pythonTuple[1].label)!
        }

        // filter out samples without annotations
        let motionSamples = motionData.motionSamples.filter { $0.annotations.count > 0 }
        
        // split into train/test sets
        let (trainMotionSamples, testMotionSamples) = motionSamples.trainTestSplit(split: 0.8)

        print("trainMotionSamples.count = \(trainMotionSamples.count)")
        print("testMotionSamples.count = \(testMotionSamples.count)")

        // samples to tensors
        self.trainingExamples = trainMotionSamples.map {
            Motion2Label2.getExample($0, labelsDict: _labelsDict, labels: _labels, tensorWidth: maxSequenceLength) 
        }.lazy.map(exampleMap)
         self.validationExamples = testMotionSamples.map {
            Motion2Label2.getExample($0, labelsDict: _labelsDict, labels: _labels, tensorWidth: maxSequenceLength)
        }.lazy.map(exampleMap)

        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        self.labels = _labels
        self.labelsDict = _labelsDict

        // Create the training sequence of epochs.
        self.trainingEpochs = TrainingEpochs(
                samples: trainingExamples, batchSize: batchSize / maxSequenceLength, entropy: entropy
            ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledMotionBatch> in
            batches.lazy.map { 
                (
                    data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
                    label: Tensor($0.map(\.label))
                )
            }
        }
        
        // Create the validation collection of batches.
        self.validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength).lazy.map { 
            (
                data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
                label: Tensor($0.map(\.label))
            )
        }
    }

    static func getExample(_ ms: MotionSample, labelsDict: [Int: String], labels: [String], tensorWidth: Int) -> Motion2LabelExample {
        let sample_id = ms.sampleID
        let labelStr: String? = labelsDict[sample_id]
        let label: Motion2LabelExample.LabelTuple? = Motion2LabelExample.LabelTuple(idx: labels.firstIndex(of: labelStr!)!, label: labelStr!)
        return Motion2LabelExample(id: "\(sample_id)", motionSample: ms, label: label)
    }
}

extension Motion2Label2 where Entropy == SystemRandomNumberGenerator {
    /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
    /// by `maximumSequenceLength`.
    ///
    /// - Parameter exampleMap: a transform that processes `Motion2LabelExample` in `LabeledMotionBatch`.
    public init(
        serializedDatasetURL: URL,
        labelsURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        exampleMap: @escaping (Motion2LabelExample) -> LabeledMotionBatch
    ) throws {
        try self.init(
            serializedDatasetURL: serializedDatasetURL,
            labelsURL: labelsURL,
            maxSequenceLength: maxSequenceLength,
            batchSize: batchSize,
            entropy: SystemRandomNumberGenerator(),
            exampleMap: exampleMap
        )
    }
}
