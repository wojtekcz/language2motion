import Foundation
import TensorFlow
import ModelSupport

public typealias LangMotionBatch = LabeledData<LangMotion.Source, LangMotion.Target>

public struct LangMotion {

    public struct Sentence {
        public var tokenIds: Tensor<Int32>   // bs x maxTextSequenceLength
        public var mask: Tensor<Float>       // bs x 1 x maxTextSequenceLength
        public let tokenCount: Tensor<Int32> // bs

        public init(tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>) {
            self.tokenIds = tokenIds
            self.mask = mask
            self.tokenCount = tokenCount
        }

        public init(copying sentence: Sentence, to device: Device) {
            tokenIds = Tensor<Int32>(copying: sentence.tokenIds, to: device)
            mask = Tensor<Float>(copying: sentence.mask, to: device)
            tokenCount = Tensor<Int32>(copying: sentence.tokenCount, to: device)
        }

        public func printSentence() {
            print("sentence")
            print("  tokenIds.shape: \(self.tokenIds.shape)")
            print("  mask.shape: \(self.mask.shape)")
            print("  tokenCount: shape \(self.tokenCount.shape), value \(self.tokenCount)")
        }
    }

    public struct MotionPart {
        public var motion: Tensor<Float>          // bs x maxMotionLength-1 x nbJoints
        public var mask: Tensor<Float>            // bs x maxMotionLength-1 x maxMotionLength-1

        public init(motion: Tensor<Float>, mask: Tensor<Float>) {
            self.motion = motion
            self.mask = mask
        }

        public init(copying motionPart: MotionPart, to device: Device) {
            motion = Tensor<Float>(copying: motionPart.motion, to: device)
            mask = Tensor<Float>(copying: motionPart.mask, to: device)
        }

        public func printMotionPart() {
            print("motionPart")
            print("  motion.shape: \(self.motion.shape)")
            print("  mask.shape: \(self.mask.shape)")
        }
    }

    public struct Source {
        public var sentence: Sentence
        public var motionPart: MotionPart

        public init(sentence: Sentence, motionPart: MotionPart) {
            self.sentence = sentence
            self.motionPart = motionPart
        }

        public init(copying source: Source, to device: Device) {
            sentence = Sentence(copying: source.sentence, to: device)
            motionPart = MotionPart(copying: source.motionPart, to: device)
        }

        public func printSource() {
            print("source")
            print("  sentence")
            print("    tokenIds.shape: \(self.sentence.tokenIds.shape)")
            print("    mask.shape: \(self.sentence.mask.shape)")
            print("    tokenCount: shape \(self.sentence.tokenCount.shape), value \(self.sentence.tokenCount)")
            print("  motionPart")
            print("    motion.shape: \(self.motionPart.motion.shape)")
            print("    mask.shape: \(self.motionPart.mask.shape)")
        }
    }

    // source
    // (padded)
    public let source: Source

    // Target
    public struct Target {
        public let sampleID: Tensor<Int32>        // bs

        // TODO: motion truth should be a STRUCT with motion and stop components
        public var targetTruth: Tensor<Float>           // bs x maxMotionLength-1 x nbJoints
        public var targetTruthStop: Tensor<Float>       // bs x maxMotionLength-1
        public let origMotionFramesCount: Tensor<Int32> // bs

        public init(sampleID: Tensor<Int32>, targetTruth: Tensor<Float>, targetTruthStop: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
            self.sampleID = sampleID

            self.targetTruth = targetTruth
            self.targetTruthStop = targetTruthStop
            self.origMotionFramesCount = origMotionFramesCount
        }

        public init(copying target: Target, to device: Device) {
            sampleID = Tensor<Int32>(copying: target.sampleID, to: device)

            targetTruth = Tensor<Float>(copying: target.targetTruth, to: device)
            targetTruthStop = Tensor<Float>(copying: target.targetTruthStop, to: device)
            origMotionFramesCount = Tensor<Int32>(copying: target.origMotionFramesCount, to: device)
        }

        public func printTarget2() {
            print("target")
            print("  sampleID.shape: \(self.sampleID.shape)")

            print("  targetTruth.shape: \(self.targetTruth.shape)")
            print("  targetTruthStop.shape: \(self.targetTruthStop.shape)")
            print("  origMotionFramesCount.shape: \(self.origMotionFramesCount.shape)")
        }
    }

    // target
    // (padded)
    public var target: Target

    public init(sampleID: Tensor<Int32>, 
                tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>, 
                motionPartTensor: Tensor<Float>, motionPartMask: Tensor<Float>,
                targetTruth: Tensor<Float>, targetTruthStop: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
        let motionPart = MotionPart(motion: motionPartTensor, mask: motionPartMask)
        let sentence = Sentence(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        self.source = Source(sentence: sentence, motionPart: motionPart)
        self.target = Target(sampleID: sampleID, targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
    }
}

public func subsequentMask3(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}

extension LangMotionBatch {
    public typealias Source = LangMotion.Source
    public typealias Sentence = LangMotion.Sentence
    public typealias MotionPart = LangMotion.MotionPart
    public typealias Target = LangMotion.Target

    public init(copying batch: LangMotionBatch, to device: Device) {
        let data = Source(copying: batch.data, to: device)
        let label = Target(copying: batch.label, to: device)
        self.init(data: data, label: label)
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenIds.squeezingShape(at: 0) })
        let mask: Tensor<Float> = Tensor(batches.map{ $0.data.sentence.mask.squeezingShape(at: 0) })
        let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenCount.squeezingShape(at: 0) })
        let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.motion.squeezingShape(at: 0) })
        let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.mask.squeezingShape(at: 0) })

        let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.label.sampleID.squeezingShape(at: 0) })
        let targetTruth: Tensor<Float> = Tensor(batches.map{ $0.label.targetTruth.squeezingShape(at: 0) })
        let targetTruthStop: Tensor<Float> = Tensor(batches.map{ $0.label.targetTruthStop.squeezingShape(at: 0) })
        let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.label.origMotionFramesCount.squeezingShape(at: 0) })

        let sentence = Sentence(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        let motionPart = MotionPart(motion: motionPartTensor, mask: motionPartMask)
        let data = Source(sentence: sentence, motionPart: motionPart)
        let label = Target(sampleID: sampleID, targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
        let batch = LangMotionBatch(data: data,label: label)

        return batch
    }

    public static func preprocessTargetMotion(sampleID: Int, motion: Tensor<Float>, maxMotionLength: Int) -> (target: Target, motionPart: MotionPart)
    {
        var (motion, motionFlag) = Tensor<Float>(motion).paddedAndCropped(to: maxMotionLength)
        motion = motion.expandingShape(at: 0)
        motionFlag = motionFlag.expandingShape(at: 0)
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(motion.shape[0])])

        let rangeExceptLast = 0..<(motion.shape[1] - 1)
        let targetMotion = motion[0..., rangeExceptLast, 0...]

        motionFlag = motionFlag[0..., rangeExceptLast]
        let targetMask = makeStandardMask(target: motionFlag, pad: 0)
        let targetTruth: Tensor<Float> = motion[0..., 1..., 0...]

        // FIXME: should targetTruthStop encompass current motion frame?
        let targetTruthStop: Tensor<Float> = 1.0 - Tensor<Float>(motionFlag)

        let motionPart = MotionPart(motion: targetMotion, mask: targetMask)
        let target = Target(sampleID: Tensor([Int32(sampleID)]), targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)

        return (target: target, motionPart: motionPart)
    }

    public static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask3(size: target.shape.last!)
        return Tensor<Float>(targetMask)
    }
}
