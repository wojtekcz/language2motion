import Foundation
import ModelSupport
import TensorFlow
import PythonKit

public typealias LangMotionBatch2 = LabeledData<LangMotionBatch.Source, LangMotionBatch.Target>

public struct Lang2Motion {

    public struct LangRec {
        public let sampleID: Int
        public let text: String
    }

    // rename MotionSample and related structs to Legacy and MotionSample to MotionSample...
    public let motionDataset: MotionDataset
    public let scaler: Scaler

    public let motionSamples: [MotionSample]
    public let langRecs: [LangRec]

    public let motionSampleDict: [Int: MotionSample]

    public let trainMotionSamples: [MotionSample]
    public let testMotionSamples: [MotionSample]

    public typealias LazySamples = LazyMapSequence<[MotionSample], LangMotionBatch2>

    public let batchSize: Int

    /// The type of the collection of batches.
    public typealias Batches = Slices<Sampling<LazySamples, ArraySlice<Int>>>
    /// The type of the training sequence of epochs.
    public typealias TrainEpochs = LazyMapSequence<TrainingEpochs<LazySamples, SystemRandomNumberGenerator>, 
        LazyMapSequence<Batches, LangMotionBatch2>>
    /// The sequence of training data (epochs of batches).
    public var trainEpochs: TrainEpochs
    /// The test batches.
    public var testBatches: LazyMapSequence<Slices<LazySamples>, LangMotionBatch2>
}

extension Lang2Motion {

    public init(
        motionDatasetURL: URL,
        batchSize: Int,
        minMotionLength: Int = 10,
        trainTestSplit: Double = 0.8,
        exampleMap: @escaping (MotionSample) -> LangMotionBatch2
    ) throws {
        // Load the data files.
        motionDataset = MotionDataset(from: motionDatasetURL)
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
        var _motionSamplesWithDistinctAnnotations: [MotionSample] = []

        for ms in _motionSamples {
            let samples = ms.annotations.map { (ann: String) -> MotionSample in
                MotionSample(sampleID: ms.sampleID, annotations: [ann], jointNames: ms.jointNames, timesteps: ms.timesteps, motion: ms.motion) 
            }
            _motionSamplesWithDistinctAnnotations.append(contentsOf: samples)
        }
        print("Having \(_motionSamplesWithDistinctAnnotations.count) annotations with motions")

        motionSamples = _motionSamplesWithDistinctAnnotations

        // create LangRecs
        langRecs = _motionSamplesWithDistinctAnnotations.map { LangRec(sampleID: $0.sampleID, text: $0.annotations[0]) }

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
        (_trainMotionSamples, _testMotionSamples) = _motionSamplesWithDistinctAnnotations.trainTestSplitMotionSamples(split: trainTestSplit)
        trainMotionSamples = _trainMotionSamples
        testMotionSamples = _testMotionSamples

        let trainSamples: LazySamples = _trainMotionSamples.lazy.map(exampleMap)
        let testSamples: LazySamples = _testMotionSamples.lazy.map(exampleMap)

        self.batchSize = batchSize

        // Create the training sequence of epochs.
        let entropy = SystemRandomNumberGenerator()
        trainEpochs = TrainingEpochs(
        samples: trainSamples, batchSize: batchSize, entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, LangMotionBatch2> in
            batches.lazy.map{ 
                Lang2Motion.reduceDataBatches(Array($0))
            }
        }
        
        // Create the test collection of batches.
        testBatches = testSamples.inBatches(of: batchSize).lazy.map{ 
            Lang2Motion.reduceDataBatches(Array($0))
        }
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.source.sentence.tokenIds.squeezingShape(at: 0) })
        let mask: Tensor<Float> = Tensor(batches.map{ $0.source.sentence.mask.squeezingShape(at: 0) })
        let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.source.sentence.tokenCount.squeezingShape(at: 0) })
        let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.source.motionPart.motion.squeezingShape(at: 0) })
        let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.source.motionPart.mask.squeezingShape(at: 0) })

        let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.target.sampleID.squeezingShape(at: 0) })
        let targetTruth: Tensor<Float> = Tensor(batches.map{ $0.target.targetTruth.squeezingShape(at: 0) })
        let targetTruthStop: Tensor<Float> = Tensor(batches.map{ $0.target.targetTruthStop.squeezingShape(at: 0) })
        let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.target.origMotionFramesCount.squeezingShape(at: 0) })

        let batch = LangMotionBatch(sampleID: sampleID, 
                tokenIds: tokenIds, mask: mask, tokenCount: tokenCount, 
                motionPartTensor: motionPartTensor, motionPartMask: motionPartMask,
                targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
        return batch
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch2]) -> LangMotionBatch2 {
        let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenIds.squeezingShape(at: 0) })
        let mask: Tensor<Float> = Tensor(batches.map{ $0.data.sentence.mask.squeezingShape(at: 0) })
        let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenCount.squeezingShape(at: 0) })
        let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.motion.squeezingShape(at: 0) })
        let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.mask.squeezingShape(at: 0) })

        let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.label.sampleID.squeezingShape(at: 0) })
        let targetTruth: Tensor<Float> = Tensor(batches.map{ $0.label.targetTruth.squeezingShape(at: 0) })
        let targetTruthStop: Tensor<Float> = Tensor(batches.map{ $0.label.targetTruthStop.squeezingShape(at: 0) })
        let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.label.origMotionFramesCount.squeezingShape(at: 0) })

        let sentence = LangMotionBatch.Sentence(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        let motionPart = LangMotionBatch.MotionPart(motion: motionPartTensor, mask: motionPartMask)
        let data = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
        let label = LangMotionBatch.Target(sampleID: sampleID, targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
        let batch = LangMotionBatch2(data: data,label: label)

        return batch
    }
}
