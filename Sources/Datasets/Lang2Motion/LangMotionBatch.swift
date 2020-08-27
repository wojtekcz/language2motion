import Foundation
import TensorFlow
import ModelSupport

public struct LangMotionBatch: KeyPathIterable {

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

    // TODO: rename Target2 to Target
    // Target2
    public struct Target2 {
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

        public init(copying target2: Target2, to device: Device) {
            sampleID = Tensor<Int32>(copying: target2.sampleID, to: device)

            targetTruth = Tensor<Float>(copying: target2.targetTruth, to: device)
            targetTruthStop = Tensor<Float>(copying: target2.targetTruthStop, to: device)
            origMotionFramesCount = Tensor<Int32>(copying: target2.origMotionFramesCount, to: device)
        }

        public func printTarget2() {
            print("target2")
            print("  sampleID.shape: \(self.sampleID.shape)")

            print("  targetTruth.shape: \(self.targetTruth.shape)")
            print("  targetTruthStop.shape: \(self.targetTruthStop.shape)")
            print("  origMotionFramesCount.shape: \(self.origMotionFramesCount.shape)")
        }
    }

    // target2
    // (padded)
    public var target2: Target2

    public init(sampleID: Tensor<Int32>, 
                tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>, 
                motionPartTensor: Tensor<Float>, motionPartMask: Tensor<Float>,
                targetTruth: Tensor<Float>, targetTruthStop: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
        let motionPart = MotionPart(motion: motionPartTensor, mask: motionPartMask)
        let sentence = Sentence(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        self.source = Source(sentence: sentence, motionPart: motionPart)
        self.target2 = Target2(sampleID: sampleID, targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)
    }

    // TODO: unpack this
    // public init(sampleID: Int, source: Source, targetMotion: Tensor<Float>, maxMotionLength: Int) {
    //     let (target2, motionPart) = Self.preprocessTargetMotion(sampleID: Tensor([Int32(sampleID)]), motion: targetMotion, maxMotionLength: maxMotionLength)
    //     self.source = source
    //     self.target2 = target2
    // }

    public static func preprocessTargetMotion(sampleID: Int, motion: Tensor<Float>, maxMotionLength: Int) -> (target2: Target2, motionPart: MotionPart)
    {
        var (motion, motionFlag) = Tensor<Float>(motion).paddedAndCropped(to: maxMotionLength)
        motion = motion.expandingShape(at: 0)
        motionFlag = motionFlag.expandingShape(at: 0)
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(motion.shape[0])])

        let rangeExceptLast = 0..<(motion.shape[1] - 1)
        let targetMotion = motion[0..., rangeExceptLast, 0...]

        motionFlag = motionFlag[0..., rangeExceptLast]
        let targetMask = LangMotionBatch.makeStandardMask(target: motionFlag, pad: 0)
        let targetTruth: Tensor<Float> = motion[0..., 1..., 0...]

        // FIXME: should targetTruthStop encompass current motion frame?
        let targetTruthStop: Tensor<Float> = 1.0 - Tensor<Float>(motionFlag)

        let motionPart = MotionPart(motion: targetMotion, mask: targetMask)
        let target2 = LangMotionBatch.Target2(sampleID: Tensor([Int32(sampleID)]), targetTruth: targetTruth, targetTruthStop: targetTruthStop, origMotionFramesCount: origMotionFramesCount)

        return (target2: target2, motionPart: motionPart)
    }

    public static func makeStandardMask(target: Tensor<Int32>, pad: Int32) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask3(size: target.shape.last!)
        return Tensor<Float>(targetMask)
    }
}

public func subsequentMask3(size: Int) -> Tensor<Int32> {
    let attentionShape = [1, size, size]
    return Tensor<Int32>(ones: TensorShape(attentionShape))
        .bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
}

extension LangMotionBatch {
    public init(copying batch: LangMotionBatch, to device: Device) {
        self.source = Source(copying: batch.source, to: device)
        self.target2 = Target2(copying: batch.target2, to: device)
    }
}

extension LangMotionBatch {
    public func printBatch() {
        print("source")
        print("  sentence")
        print("    tokenIds.shape: \(self.source.sentence.tokenIds.shape)")
        print("    mask.shape: \(self.source.sentence.mask.shape)")
        print("    tokenCount: shape \(self.source.sentence.tokenCount.shape), value \(self.source.sentence.tokenCount)")
        print("  motionPart")
        print("    motion.shape: \(self.source.motionPart.motion.shape)")
        print("    mask.shape: \(self.source.motionPart.mask.shape)")

        print("target2")
        print("  sampleID: shape \(self.target2.sampleID.shape), value \(self.target2.sampleID)")
        print("  targetTruth.shape: \(self.target2.targetTruth.shape)")
        print("  targetTruthStop.shape: \(self.target2.targetTruthStop.shape)")
        print("  origMotionFramesCount: shape \(self.target2.origMotionFramesCount.shape), value \(self.target2.origMotionFramesCount)")
    }
}

extension LangMotionBatch2 {
    public init(copying batch: LangMotionBatch2, to device: Device) {
        let data = LangMotionBatch.Source(copying: batch.data, to: device)
        let label = LangMotionBatch.Target2(copying: batch.label, to: device)
        self.init(data: data, label: label)
    }
}
