import Foundation
import TensorFlow

public struct LangMotionBatch: KeyPathIterable {
    public let sampleID: Tensor<Int32>   // bs

    public struct Source {
        public var tokenIds: Tensor<Int32>   // bs x maxTextSequenceLength
        public var mask: Tensor<Float>       // bs x 1 x maxTextSequenceLength
        public let tokenCount: Tensor<Int32> // bs

        public init(tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>) {
            self.tokenIds = tokenIds
            self.mask = mask
            self.tokenCount = tokenCount
        }

        public init(copying source: Source, to device: Device) {
            tokenIds = Tensor<Int32>(copying: source.tokenIds, to: device)
            mask = Tensor<Float>(copying: source.mask, to: device)
            tokenCount = Tensor<Int32>(copying: source.tokenCount, to: device)
        }

        public func printSource() {
            print("source")
            print("  tokenIds.shape: \(self.tokenIds.shape)")
            print("  mask.shape: \(self.mask.shape)")
            print("  tokenCount: shape \(self.tokenCount.shape), value \(self.tokenCount)")
        }
    }

    // source
    // (padded)
    public let source: Source
    
    public struct Target {
        public var motion: Tensor<Float>          // bs x maxMotionLength-1 x nbJoints
        public var mask: Tensor<Float>            // bs x maxMotionLength-1 x maxMotionLength-1

        public init(motion: Tensor<Float>, mask: Tensor<Float>) {
            self.motion = motion
            self.mask = mask
        }

        public init(copying target: Target, to device: Device) {
            motion = Tensor<Float>(copying: target.motion, to: device)
            mask = Tensor<Float>(copying: target.mask, to: device)
        }

        public func printTarget() {
            print("target")
            print("  motion.shape: \(self.motion.shape)")
            print("  mask.shape: \(self.mask.shape)")
        }
    }

    // target
    // (padded)
    public var target: Target

    public var targetTruth: Tensor<Float>           // bs x maxMotionLength-1 x nbJoints
    public var targetTruthStop: Tensor<Float>       // bs x maxMotionLength-1
    // TODO: target truth should be a STRUCT with motion and stop components
    // TODO: how targetMask is used and consumed?
    // used by transformer decoder? by mixture model?

    public let origMotionFramesCount: Tensor<Int32> // bs

    public init(sampleID: Tensor<Int32>, 
                tokenIds: Tensor<Int32>, mask: Tensor<Float>, tokenCount: Tensor<Int32>, 
                targetMotion: Tensor<Float>, targetMask: Tensor<Float>,
                targetTruth: Tensor<Float>, targetTruthStop: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
        self.sampleID = sampleID
        self.source = Source(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        self.target = Target(motion: targetMotion, mask: targetMask)

        self.targetTruth = targetTruth
        self.targetTruthStop = targetTruthStop
        self.origMotionFramesCount = origMotionFramesCount
    }

    public init(sampleID: Tensor<Int32>, source: Source, target: Target,
                targetTruth: Tensor<Float>, targetTruthStop: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
        self.sampleID = sampleID
        self.source = source

        self.target = target
        self.targetTruth = targetTruth
        self.targetTruthStop = targetTruthStop
        self.origMotionFramesCount = origMotionFramesCount
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
        self.sampleID = Tensor<Int32>(copying: batch.sampleID, to: device)
        self.source = Source(copying: batch.source, to: device)
        self.target = Target(copying: batch.target, to: device)

        self.targetTruth = Tensor<Float>(copying: batch.targetTruth, to: device)
        self.targetTruthStop = Tensor<Float>(copying: batch.targetTruthStop, to: device)
        self.origMotionFramesCount = Tensor<Int32>(copying: batch.origMotionFramesCount, to: device)
    }
}

extension LangMotionBatch {
    public func printBatch() {
        print("sampleID: shape \(self.sampleID.shape), value \(self.sampleID)")

        print("source")
        print("  tokenIds.shape: \(self.source.tokenIds.shape)")
        print("  mask.shape: \(self.source.mask.shape)")
        print("  tokenCount: shape \(self.source.tokenCount.shape), value \(self.source.tokenCount)")

        print("target")
        print("  targetMotion.shape: \(self.target.motion.shape)")
        print("  targetMask.shape: \(self.target.mask.shape)")
        print("  targetTruth.shape: \(self.targetTruth.shape)")
        print("  targetTruthStop.shape: \(self.targetTruthStop.shape)")
        print("  origMotionFramesCount: shape \(self.origMotionFramesCount.shape), value \(self.origMotionFramesCount)")
    }
}
