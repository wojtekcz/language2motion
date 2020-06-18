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


public struct Motion2Label <Entropy: RandomNumberGenerator> {
    public typealias Samples = LazyMapSequence<[Motion2LabelExample], LabeledMotionBatch>

    public let motionDataset: MotionDataset
    public let testMotionSamples: [MotionSample]
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

extension Motion2Label {
    static func filterSamples(_ motionSamples: [MotionSample], classIdx: Int, labelsDict: [Int: String], labels: [String]) -> [MotionSample] {
        let motionSamplesForClass = motionSamples.filter {
            (ms: MotionSample) -> Bool in
            let labelTuple = Motion2Label.getLabel(sampleID: ms.sampleID, labelsDict: labelsDict, labels: labels)!
            return labelTuple.idx == classIdx
        }
        return motionSamplesForClass
    }

    static func balanceClassSamples(_ motionSamples: [MotionSample], numPerClass: Int, split: Double = 0.8, labelsDict: [Int: String], labels: [String]) -> (trainSamples: [MotionSample], testSamples: [MotionSample]) {
        var allTrainSamples: [MotionSample] = []
        var allTestSamples: [MotionSample] = []

        for classIdx in (0..<labels.count) { 
            let samplesForClass = Motion2Label.filterSamples(motionSamples, classIdx: classIdx, labelsDict: labelsDict, labels: labels)

            var trainSamples: [MotionSample]
            var testSamples: [MotionSample]
            if samplesForClass.count >= numPerClass { // downsample
                let sampledSamplesForClass = Array(samplesForClass.choose(numPerClass))
                (trainSamples, testSamples) = sampledSamplesForClass.trainTestSplitMotionSamples(split: split)
            } else { // upsample
                (trainSamples, testSamples) = samplesForClass.trainTestSplitMotionSamples(split: split)
                let maxTrainPerClass = Int(Double(numPerClass)*split)
                trainSamples = (0..<maxTrainPerClass).map { (a) -> MotionSample in trainSamples.randomElement()! }
            }

            allTrainSamples.append(contentsOf: trainSamples)
            allTestSamples.append(contentsOf: testSamples)

            print((samplesForClass.count, trainSamples.count, testSamples.count))
        }
        allTrainSamples.shuffle()
        allTestSamples.shuffle()
        return (trainSamples: allTrainSamples, testSamples: allTestSamples)
    }
}

extension Motion2Label {

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
        balanceClassSamples: Int? = nil,
        trainTestSplit: Double = 0.8,
        entropy: Entropy,
        exampleMap: @escaping (Motion2LabelExample) -> LabeledMotionBatch
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset(from: serializedDatasetURL)
        print(motionDataset.description)
        
        let df = pd.read_csv(labelsURL.path)
        let _labels = df.label.unique().sorted().map { String($0)! }

        var _labelsDict: [Int: String] = [:]
        for pythonTuple in df.iterrows() {
            _labelsDict[Int(pythonTuple[1].sample_id)!] = String(pythonTuple[1].label)!
        }

        // filter out samples without annotations
        var motionSamples = motionDataset.motionSamples.filter { $0.annotations.count > 0 }
        print("keeping \(motionSamples.count) annotatated motions")

        // filter out shortest samples
        let minMotionLength = 10 // 1 sec. (for downsampled motion)
        motionSamples = motionSamples.filter { $0.motionFramesArray.shape[0] >= minMotionLength }
        print("keeping \(motionSamples.count) longer motions, with minimum \(minMotionLength) frames")
        
        // split into train/test sets
        var trainMotionSamples: [MotionSample] = []
        var testMotionSamples: [MotionSample] = []
        if balanceClassSamples == nil {
            (trainMotionSamples, testMotionSamples) = motionSamples.trainTestSplitMotionSamples(split: trainTestSplit)
        } else {
            print("\nClass balancing...")
            (trainMotionSamples, testMotionSamples) = Motion2Label.balanceClassSamples(
                motionSamples, numPerClass: balanceClassSamples!, split: trainTestSplit, labelsDict: _labelsDict, labels: _labels
            )
        }
        self.testMotionSamples = testMotionSamples

        // samples to tensors
        self.trainingExamples = trainMotionSamples.map {
            Motion2Label.getExample($0, labelsDict: _labelsDict, labels: _labels, tensorWidth: maxSequenceLength) 
        }.lazy.map(exampleMap)
         self.validationExamples = testMotionSamples.map {
            Motion2Label.getExample($0, labelsDict: _labelsDict, labels: _labels, tensorWidth: maxSequenceLength)
        }.lazy.map(exampleMap)

        self.maxSequenceLength = maxSequenceLength
        self.batchSize = batchSize

        self.labels = _labels
        self.labelsDict = _labelsDict

        // Create the training sequence of epochs.
        self.trainingEpochs = TrainingEpochs(
                samples: trainingExamples, batchSize: batchSize, entropy: entropy
            ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LabeledMotionBatch> in
                batches.lazy.map { 
                    LabeledMotionBatch(
                        data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
                        label: Tensor($0.map(\.label))
                    )
                }
        }
        
        // Create the validation collection of batches.
        self.validationBatches = validationExamples.inBatches(of: batchSize).lazy.map { 
            LabeledMotionBatch(
                data: $0.map(\.data).paddedAndCollated(to: maxSequenceLength),
                label: Tensor($0.map(\.label))
            )
        }
    }

    static func getLabel(sampleID: Int, labelsDict: [Int: String], labels: [String]) -> Motion2LabelExample.LabelTuple? {
        let labelStr: String? = labelsDict[sampleID]
        
        var label: Motion2LabelExample.LabelTuple? = nil
        if labelStr != nil {
            label = Motion2LabelExample.LabelTuple(idx: labels.firstIndex(of: labelStr!)!, label: labelStr!)
        }
        return label
    }

    public func getLabel(_ sampleID: Int) -> Motion2LabelExample.LabelTuple? {
        return Motion2Label.getLabel(sampleID: sampleID, labelsDict: labelsDict, labels: labels)
    }

    static func getExample(_ ms: MotionSample, labelsDict: [Int: String], labels: [String], tensorWidth: Int) -> Motion2LabelExample {
        let label: Motion2LabelExample.LabelTuple? = Motion2Label.getLabel(sampleID: ms.sampleID, labelsDict: labelsDict, labels: labels)
        return Motion2LabelExample(id: "\(ms.sampleID)", motionSample: ms, label: label)
    }
}

extension Motion2Label where Entropy == SystemRandomNumberGenerator {
    /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
    /// by `maximumSequenceLength`.
    ///
    /// - Parameter exampleMap: a transform that processes `Motion2LabelExample` in `LabeledMotionBatch`.
    public init(
        serializedDatasetURL: URL,
        labelsURL: URL,
        maxSequenceLength: Int,
        batchSize: Int,
        balanceClassSamples: Int? = nil,
        trainTestSplit: Double = 0.8,
        exampleMap: @escaping (Motion2LabelExample) -> LabeledMotionBatch
    ) throws {
        try self.init(
            serializedDatasetURL: serializedDatasetURL,
            labelsURL: labelsURL,
            maxSequenceLength: maxSequenceLength,
            batchSize: batchSize,
            balanceClassSamples: balanceClassSamples,
            trainTestSplit: trainTestSplit,
            entropy: SystemRandomNumberGenerator(),
            exampleMap: exampleMap
        )
    }
}
