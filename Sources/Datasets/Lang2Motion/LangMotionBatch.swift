import Foundation
import TensorFlow
import ModelSupport

public typealias LangMotionBatch = LabeledData<LangMotion.Source, LangMotion.Target>

public struct LangMotion {

    public struct Sentence {
        public var tokenIds: Tensor<Int32>   // bs x maxTextSequenceLength
        public var mask: Tensor<Float>       // bs x maxMotionLength x maxTextSequenceLength
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

    // FIXME: -1 is obsolete?
    public struct MotionPart {
        public var motion: Tensor<Float>          // bs x maxMotionLength-1 x nbJoints

        // self-attention mask
        public var mask: Tensor<Float>            // bs x maxMotionLength-1 x maxMotionLength-1 // FIXME: should be bs x maxMotionLength x maxTextSequenceLength?
        public var startFlag: Tensor<Float>       // bs x maxMotionLength-1 x 1
        public var motionFlag: Tensor<Int32>      // bs x maxMotionLength-1 x 1

        public init(motion: Tensor<Float>, mask: Tensor<Float>, startFlag: Tensor<Float>, motionFlag: Tensor<Int32>) {
            self.motion = motion
            self.mask = mask
            self.startFlag = startFlag
            self.motionFlag = motionFlag

            assert(startFlag.shape.count == 3 && startFlag.shape[2]==1)
            assert(motionFlag.shape.count == 3 && motionFlag.shape[2]==1)
        }

        public init(copying motionPart: MotionPart, to device: Device) {
            motion = Tensor<Float>(copying: motionPart.motion, to: device)
            mask = Tensor<Float>(copying: motionPart.mask, to: device)
            startFlag = Tensor<Float>(copying: motionPart.startFlag, to: device)
            motionFlag = Tensor<Int32>(copying: motionPart.motionFlag, to: device)
        }

        public func printMotionPart() {
            print("motionPart")
            print("  motion.shape: \(self.motion.shape)")
            print("  mask.shape: \(self.mask.shape)")
            print("  startFlag.shape: \(self.startFlag.shape)")
            print("  motionFlag.shape: \(self.motionFlag.shape)")
        }
    }

    public struct Source {
        public var sentence: Sentence
        public var motionPart: MotionPart
        public var sourceAttentionMask: Tensor<Float>       // bs x maxMotionLength x maxTextSequenceLength

        public init(sentence: Sentence, motionPart: MotionPart, sourceAttentionMask: Tensor<Float>) {
            self.sentence = sentence
            self.motionPart = motionPart
            self.sourceAttentionMask = sourceAttentionMask
        }

        public init(sentence: Sentence, motionPart: MotionPart) {
            let sentenceMask = sentence.mask.squeezingShape(at: 1)
            let motionFlag = Tensor<Float>(motionPart.motionFlag).squeezingShape(at: 2)
            let sourceAttentionMask = (sentenceMask * motionFlag.transposed()).expandingShape(at: 0)
            self.init(sentence: sentence, motionPart: motionPart, sourceAttentionMask: sourceAttentionMask)
        }

        public init(copying source: Source, to device: Device) {
            sentence = Sentence(copying: source.sentence, to: device)
            motionPart = MotionPart(copying: source.motionPart, to: device)
            sourceAttentionMask = Tensor(copying: source.sourceAttentionMask, to: device)
        }

        public func printSource() {
            print("source")
            print("  sentence")
            print("    tokenIds.shape: \(self.sentence.tokenIds.shape)")
            print("    mask.shape: \(self.sentence.mask.shape)")
            print("    tokenCount: (shape: \(self.sentence.tokenCount.shape), value: \(self.sentence.tokenCount))")
            print("  motionPart")
            print("    motion.shape: \(self.motionPart.motion.shape)")
            print("    mask.shape: \(self.motionPart.mask.shape)")
            print("    startFlag.shape: \(self.motionPart.startFlag.shape)")
            print("    motionFlag.shape: \(self.motionPart.motionFlag.shape)")
            print("  sourceAttentionMask.shape: \(self.sourceAttentionMask.shape)")
        }
    }

    public struct Target {
        public let sampleID: Tensor<Int32>        // bs

        // FIXME: -1 is obsolete?
        public var motion: Tensor<Float>          // bs x maxMotionLength-1 x nbJoints
        public var stops: Tensor<Float>           // bs x maxMotionLength-1

        public let origMotionFramesCount: Tensor<Int32> // bs

        public init(sampleID: Tensor<Int32>, motion: Tensor<Float>, stops: Tensor<Float>, origMotionFramesCount: Tensor<Int32>) {
            self.sampleID = sampleID

            self.motion = motion
            self.stops = stops
            self.origMotionFramesCount = origMotionFramesCount
        }

        public init(copying target: Target, to device: Device) {
            sampleID = Tensor<Int32>(copying: target.sampleID, to: device)

            motion = Tensor<Float>(copying: target.motion, to: device)
            stops = Tensor<Float>(copying: target.stops, to: device)
            origMotionFramesCount = Tensor<Int32>(copying: target.origMotionFramesCount, to: device)
        }

        public func printTarget() {
            print("target")
            print("  sampleID: (shape: \(self.sampleID.shape), value: \(self.sampleID))")

            print("  motion.shape: \(self.motion.shape)")
            print("  stops.shape: \(self.stops.shape)")
            print("  origMotionFramesCount: (shape: \(self.origMotionFramesCount.shape), value: \(self.origMotionFramesCount))")
        }
    }
}

extension LangMotionBatch {
    public typealias Source = LangMotion.Source
    public typealias Sentence = LangMotion.Sentence
    public typealias MotionPart = LangMotion.MotionPart
    public typealias Target = LangMotion.Target

    public var source: Source { get { return data } }
    public var target: Target { get { return label } }

    public init(source: Source, target: Target) {
        self.init(data: source, label: target)
    }

    public init(copying batch: LangMotionBatch, to device: Device) {
        let data = Source(copying: batch.data, to: device)
        let label = Target(copying: batch.label, to: device)
        self.init(data: data, label: label)
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        // TODO: refactor: move code into sub-structures
        let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenIds.squeezingShape(at: 0) })
        let mask: Tensor<Float> = Tensor(batches.map{ $0.data.sentence.mask.squeezingShape(at: 0) })
        let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.data.sentence.tokenCount.squeezingShape(at: 0) })
        let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.motion.squeezingShape(at: 0) })
        let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.mask.squeezingShape(at: 0) })
        let motionPartStartFlag: Tensor<Float> = Tensor(batches.map{ $0.data.motionPart.startFlag.squeezingShape(at: 0) })
        let motionPartFlag: Tensor<Int32> = Tensor(batches.map{ $0.data.motionPart.motionFlag.squeezingShape(at: 0) })
        let sourceAttentionMask: Tensor<Float> = Tensor(batches.map{ $0.data.sourceAttentionMask.squeezingShape(at: 0) })

        let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.label.sampleID.squeezingShape(at: 0) })
        let targetMotion: Tensor<Float> = Tensor(batches.map{ $0.label.motion.squeezingShape(at: 0) })
        let targetStops: Tensor<Float> = Tensor(batches.map{ $0.label.stops.squeezingShape(at: 0) })
        let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.label.origMotionFramesCount.squeezingShape(at: 0) })

        let sentence = Sentence(tokenIds: tokenIds, mask: mask, tokenCount: tokenCount)
        let motionPart = MotionPart(motion: motionPartTensor, mask: motionPartMask, startFlag: motionPartStartFlag, motionFlag: motionPartFlag)
        let data = Source(sentence: sentence, motionPart: motionPart, sourceAttentionMask: sourceAttentionMask)
        let label = Target(sampleID: sampleID, motion: targetMotion, stops: targetStops, origMotionFramesCount: origMotionFramesCount)
        let batch = LangMotionBatch(data: data,label: label)

        return batch
    }

    // TODO: kill startMotionToken?
    public static func startMotionToken(nbJoints: Int) -> Tensor<Float> {
        // one motion frame [1, nbJoints]
        return Tensor<Float>(repeating:1.0, shape: [1, nbJoints])
    }

    public static func zeroMotionFrame(nbJoints: Int) -> Tensor<Float> {
        // one motion frame [1, nbJoints]
        return Tensor<Float>(zeros: [1, nbJoints])
    }

    public static func preprocessTargetMotion(sampleID: Int, motion: Tensor<Float>, maxMotionLength: Int, shiftMaskRight: Bool = false) -> (motionPart: MotionPart, target: Target)
    {
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(motion.shape[0])])
        let nbJoints = motion.shape[1]
        
        let zeroMotionFrame = Self.zeroMotionFrame(nbJoints: nbJoints)
        
        let motion2 = Tensor(concatenating: [zeroMotionFrame, motion], alongAxis: 0)

        var (paddedMotion, motionFlag) = motion2.paddedAndCropped(to: maxMotionLength+1)
        paddedMotion = paddedMotion.expandingShape(at: 0) // FIXME: move adding batch dimension further down
        motionFlag = motionFlag.expandingShape(at: 0) // FIXME: move adding batch dimension further down

        // source (motionPart & motion flag)
        let rangeExceptLast = 0..<(paddedMotion.shape[1] - 1)
        let motionPartTensor = paddedMotion[0..., rangeExceptLast, 0...]

        let motionPartFlag = motionFlag[0..., rangeExceptLast]
        var motionPartMask = makeStandardMask(target: motionPartFlag, pad: 0, shiftRight: shiftMaskRight) // FIXME: fix target mask
        let motionLen = Int(motionFlag.sum().scalar!)
        motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
        motionPartMask = abs(motionPartMask)

        var motionStartFlag = Tensor<Float>(zeros: [motionPartTensor.shape[1], 1]).expandingShape(at: 0)
        motionStartFlag[0, 0, 0] = Tensor(1.0)

        let motionPart = MotionPart(motion: motionPartTensor, mask: motionPartMask, startFlag: motionStartFlag, motionFlag: motionPartFlag.expandingShape(at: 2))

        // target (motion & stops)
        let targetMotion: Tensor<Float> = paddedMotion[0..., 1..., 0...]
        let targetMotionFlag = motionFlag[0..., 1...]
        let targetStops: Tensor<Float> = 1.0 - Tensor<Float>(targetMotionFlag)

        let target = Target(sampleID: Tensor([Int32(sampleID)]), motion: targetMotion, stops: targetStops, origMotionFramesCount: origMotionFramesCount)
        return (motionPart: motionPart, target: target)
    }

    public static func subsequentMask(size: Int, shiftRight: Bool = false) -> Tensor<Int32> {
        let attentionShape = [1, size, size]
        let ones = Tensor<Int32>(ones: TensorShape(attentionShape))
        var mask: Tensor<Int32>
        
        if !shiftRight {
            mask = ones.bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
        } else {
            // https://www.tensorflow.org/tutorials/text/transformer#masking
            mask = 1 - ones.bandPart(subdiagonalCount: -1, superdiagonalCount: 0)
        }
        return mask
    }

    public static func makeStandardMask(target: Tensor<Int32>, pad: Int32, shiftRight: Bool = false) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target)
            .replacing(with: Tensor(onesLike: target), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask(size: target.shape.last!, shiftRight: shiftRight)
        return Tensor<Float>(targetMask)
    }
}
