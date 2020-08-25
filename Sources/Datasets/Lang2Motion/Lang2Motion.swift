import Foundation
import ModelSupport
import TensorFlow
import PythonKit

public struct Lang2Motion {

    public struct LangRec {
        public let sampleID: Int
        public let text: String
    }

    public struct Example {
        public let sampleID: Int
        public let sentence: String
        public let motionSample: MotionSample2

        public init(sampleID: Int, sentence: String, motionSample: MotionSample2) {
            self.sampleID = sampleID
            self.sentence = sentence
            self.motionSample = motionSample
        }
    }

    public let motionDataset: MotionDataset2
    public let scaler: Scaler

    public let motionSamples: [MotionSample2]
    public let langRecs: [LangRec]

    public let motionSampleDict: [Int: MotionSample2]

    public let trainExamples: [Example]
    public let valExamples: [Example]

    public typealias Samples = LazyMapSequence<[Example], LangMotionBatch>
    
    public let trainingSamples: Samples
    public let validationSamples: Samples

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
    public static func getExample(motionSample: MotionSample2) -> Example {
        return Example(sampleID: motionSample.sampleID, sentence: motionSample.annotations[0], motionSample: motionSample)
    }
}

extension Lang2Motion {

    public init(
        motionDatasetURL: URL,
        batchSize: Int,
        minMotionLength: Int = 10,
        trainTestSplit: Double = 0.8,
        exampleMap: @escaping (Example) -> LangMotionBatch
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset2(from: motionDatasetURL)
        print(motionDataset.description)

        // filter out samples without annotations
        var _motionSamples = motionDataset.motionSamples.filter { $0.annotations.count > 0 }
        print("keeping \(_motionSamples.count) annotated motions")

        // filter out shortest samples
        _motionSamples = _motionSamples.filter { $0.motion.shape[0] >= minMotionLength }
        print("keeping \(_motionSamples.count) longer motions, with minimum \(minMotionLength) frames")

        // scale motions
        print("Scaling motions...")
        let motions = _motionSamples.map { $0.motion }
        let _scaler = Scaler(X: Tensor(concatenating: motions, alongAxis: 0))
        let scaledMotions = motions.map { _scaler.transform($0) }

        for idx in 0..<_motionSamples.count {
            _motionSamples[idx].motion = scaledMotions[idx]
        }
        scaler = _scaler
        print("Motions scaled.")

        // get all annotations from motionSamples
        var _motionSamplesWithDistinctAnnotations: [MotionSample2] = []

        for ms in _motionSamples {
            let samples = ms.annotations.map { (ann: String) -> MotionSample2 in
                MotionSample2(sampleID: ms.sampleID, annotations: [ann], jointNames: ms.jointNames, timesteps: ms.timesteps, motion: ms.motion) 
            }
            _motionSamplesWithDistinctAnnotations.append(contentsOf: samples)
        }
        print("Having \(_motionSamplesWithDistinctAnnotations.count) annotations with motions")

        motionSamples = _motionSamplesWithDistinctAnnotations

        // split into train/test sets
        var trainMotionSamples: [MotionSample2] = []
        var testMotionSamples: [MotionSample2] = []
        (trainMotionSamples, testMotionSamples) = _motionSamplesWithDistinctAnnotations.trainTestSplitMotionSamples(split: trainTestSplit)

        // create LangRecs
        langRecs = _motionSamplesWithDistinctAnnotations.map { LangRec(sampleID: $0.sampleID, text: $0.annotations[0]) }

        // [sampleID:MotionSample2] mapping
        var _motionSampleDict: [Int: MotionSample2] = [:]
        for ms in motionDataset.motionSamples {
            // only assign first (downsampled) sample
            if _motionSampleDict[ms.sampleID] == nil {
                _motionSampleDict[ms.sampleID] = ms
            }
        }
        motionSampleDict = _motionSampleDict

        // create Examples
        trainExamples = trainMotionSamples.map {
            Lang2Motion.getExample(motionSample: $0)
        }
        valExamples = testMotionSamples.map {
            Lang2Motion.getExample(motionSample: $0)
        }

        trainingSamples = trainExamples.lazy.map(exampleMap)
        validationSamples = valExamples.lazy.map(exampleMap)

        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainingEpochs = TrainingEpochs(
        samples: trainingSamples, batchSize: batchSize, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LangMotionBatch> in
            batches.lazy.map{ 
                Lang2Motion.reduceDataBatches(Array($0))
            }
        }
        
        // Create the validation collection of batches.
        validationBatches = validationSamples.inBatches(of: batchSize).lazy.map{ 
            Lang2Motion.reduceDataBatches(Array($0))
        }
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.sampleID.squeezingShape(at: 0) })
        let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.source.tokenIds.squeezingShape(at: 0) })
        let mask: Tensor<Float> = Tensor(batches.map{ $0.source.mask.squeezingShape(at: 0) })
        let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.source.tokenCount.squeezingShape(at: 0) })

        let targetMotion: Tensor<Float> = Tensor(batches.map{ $0.target.motion.squeezingShape(at: 0) })
        let targetMask: Tensor<Float> = Tensor(batches.map{ $0.target.mask.squeezingShape(at: 0) })
        let targetTruth: Tensor<Float> = Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) })
        let targetTruthStop: Tensor<Float> = Tensor(batches.map{ $0.targetTruthStop.squeezingShape(at: 0) })
        let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.origMotionFramesCount.squeezingShape(at: 0) })

        let batch = LangMotionBatch(sampleID: sampleID, 
                tokenIds: tokenIds, mask: mask, tokenCount: tokenCount, 
                targetMotion: targetMotion, targetMask: targetMask,
                targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
        return batch
    }
}
