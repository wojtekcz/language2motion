import Foundation
import ModelSupport
import TensorFlow
import PythonKit

public struct Lang2Motion {

    public struct LangRec {
        public let sampleID: Int
        public let text: String
        public let motionSample: MotionSample
    }

    public let motionDataset: MotionDataset
    public let scaler: MinMaxScaler
    public var discretizer: MotionDiscretizer

    public let motionSamples: [MotionSample]
    public let langRecs: [LangRec]

    public let motionSampleDict: [Int: MotionSample]

    public let trainMotionSamples: [MotionSample]
    public let testMotionSamples: [MotionSample]

    public typealias LazySamples = LazyMapSequence<[MotionSample], LangMotionBatch>

    public let batchSize: Int

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<LazySamples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<LazySamples, SystemRandomNumberGenerator>, 
        LazyMapSequence<Batches, LangMotionBatch>>
    /// The sequence of training data (epochs of batches).
    public var trainEpochs: TrainEpochs
    /// The test batches.
    public var testBatches: LazyMapSequence<Slices<LazySamples>, LangMotionBatch>
}

extension Lang2Motion {

    public init(
        motionDatasetURL: URL,
        batchSize: Int,
        minMotionLength: Int = 10,
        maxMotionLength: Int = 100,
        multiplyFactor: Int = 1,
        discretizer: inout MotionDiscretizer,
        trainTestSplit: Double = 0.8,
        device: Device,
        exampleMap: @escaping (MotionSample) -> LangMotionBatch
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset(from: motionDatasetURL)
        print(motionDataset.description)

        // filter out samples without annotations
        var _motionSamples = motionDataset.motionSamples.filter { $0.annotations.count > 0 }
        print("Keeping \(_motionSamples.count) annotated motions.")

        // filter out shortest samples
        _motionSamples = _motionSamples.filter { $0.motion.shape[0] >= minMotionLength }
        print("Keeping \(_motionSamples.count) longer motions, with minimum \(minMotionLength) frames.")

        // filter out longest samples
        _motionSamples = _motionSamples.filter { $0.motion.shape[0] <= maxMotionLength }
        print("Keeping \(_motionSamples.count) shorter motions, with maximum \(maxMotionLength) frames.")

        // scale motions
        print("Scaling motions...")
        let motions = _motionSamples.map { $0.motion }
        let _scaler = MinMaxScaler(X: Tensor(concatenating: motions, alongAxis: 0))
        let scaledMotions = motions.map { _scaler.transform($0) }
//        let scaledMotions = motions.map { $0 }

        for idx in 0..<_motionSamples.count {
            _motionSamples[idx].motion = scaledMotions[idx]
        }
        scaler = _scaler
        print("Motions scaled.")

        // discretize motions
        discretizer.fit(Tensor(concatenating: scaledMotions, alongAxis: 0))
        print("discretizing...")
        let discreteMotions = scaledMotions.map { discretizer.transform($0) }
        print("discreteMotions.count: \(discreteMotions.count)")
        // print("inversing...")
//         let inversedMotions = discreteMotions.map { discretizer.inverse_transform($0) }

        for idx in 0..<_motionSamples.count {
            _motionSamples[idx].discreteMotion = discreteMotions[idx]
        }

        // de-discretized
//        for idx in 0..<_motionSamples.count {
//            _motionSamples[idx].motion = inversedMotions[idx]
//        }
        
        self.discretizer = discretizer
        
        // get all annotations from motionSamples
        var _motionSamplesWithDistinctAnnotations: [MotionSample] = []

        for ms in _motionSamples {
            let samples = ms.annotations.map { (ann: String) -> MotionSample in
                MotionSample(sampleID: ms.sampleID, annotations: [ann], jointNames: ms.jointNames, timesteps: ms.timesteps, motion: ms.motion) 
            }
            _motionSamplesWithDistinctAnnotations.append(contentsOf: samples)
        }
        print("Having \(_motionSamplesWithDistinctAnnotations.count) annotations with motions.")

        // multiply samples by factor
        let _multipliedSamples = Array(_motionSamplesWithDistinctAnnotations.map({sample in (0..<multiplyFactor).map({ _ in sample })}).joined())
        
        print("Having \(_multipliedSamples.count) multiplied samples.")
        motionSamples = _multipliedSamples

        // create LangRecs
        langRecs = _multipliedSamples.map { LangRec(sampleID: $0.sampleID, text: $0.annotations[0], motionSample: $0) }

        // [sampleID:MotionSample] mapping
        var _motionSampleDict: [Int: MotionSample] = [:]
        for ms in motionDataset.motionSamples {
            // only assign first (downsampled) sample
            if _motionSampleDict[ms.sampleID] == nil {
                _motionSampleDict[ms.sampleID] = ms
            }
        }
        motionSampleDict = _motionSampleDict

        // split into train/test sets
        let _trainMotionSamples: [MotionSample]
        let _testMotionSamples: [MotionSample]
        (_trainMotionSamples, _testMotionSamples) = _multipliedSamples.trainTestSplitMotionSamples(split: trainTestSplit)
        trainMotionSamples = _trainMotionSamples
        testMotionSamples = _testMotionSamples

        let trainSamples: LazySamples = _trainMotionSamples.lazy.map(exampleMap)
        let testSamples: LazySamples = _testMotionSamples.lazy.map(exampleMap)

        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainEpochs = TrainingEpochs(
            samples: trainSamples, batchSize: batchSize, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LangMotionBatch> in
            batches.lazy.map{ 
                // TODO: reduceDataBatches to device directly
                LangMotionBatch(copying: LangMotionBatch.reduceDataBatches(Array($0)), to: device)
            }
        }
        
        // Create the test collection of batches.
        testBatches = testSamples.inBatches(of: batchSize).lazy.map{ 
                LangMotionBatch(copying: LangMotionBatch.reduceDataBatches(Array($0)), to: device)
        }
    }
}
