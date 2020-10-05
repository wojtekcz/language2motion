import Foundation
import TensorFlow
import ModelSupport

public typealias LangMotionBatch = LabeledData<LangMotion.Source, LangMotion.Target>

public struct LangMotion {

    public struct Sentence {
        public var tokenIds: Tensor<Int32>   // bs x maxTextSequenceLength
        public let tokenCount: Tensor<Int32> // bs
        public var selfAttentionMask: Tensor<Float> // bs x maxMotionLength x maxTextSequenceLength // or x 1?

        public init(tokenIds: Tensor<Int32>, selfAttentionMask: Tensor<Float>, tokenCount: Tensor<Int32>) {
            self.tokenIds = tokenIds
            self.selfAttentionMask = selfAttentionMask
            self.tokenCount = tokenCount
        }

        public init(copying sentence: Sentence, to device: Device) {
            tokenIds = Tensor<Int32>(copying: sentence.tokenIds, to: device)
            selfAttentionMask = Tensor<Float>(copying: sentence.selfAttentionMask, to: device)
            tokenCount = Tensor<Int32>(copying: sentence.tokenCount, to: device)
        }

        public static func reduceDataBatches(_ batches: [Sentence]) -> Sentence {
            let tokenIds: Tensor<Int32> = Tensor(batches.map{ $0.tokenIds.squeezingShape(at: 0) })
            let selfAttentionMask: Tensor<Float> = Tensor(batches.map{ $0.selfAttentionMask.squeezingShape(at: 0) })
            let tokenCount: Tensor<Int32> = Tensor(batches.map{ $0.tokenCount.squeezingShape(at: 0) })
            return Sentence(tokenIds: tokenIds, selfAttentionMask: selfAttentionMask, tokenCount: tokenCount)
        }

        public func printSentence() {
            print("sentence")
            print("  tokenIds.shape: \(self.tokenIds.shape)")
            print("  selfAttentionMask.shape: \(self.selfAttentionMask.shape)")
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

        public static func reduceDataBatches(_ batches: [MotionPart]) -> MotionPart {
            let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.motion.squeezingShape(at: 0) })
            let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.mask.squeezingShape(at: 0) })
            let motionPartStartFlag: Tensor<Float> = Tensor(batches.map{ $0.startFlag.squeezingShape(at: 0) })
            let motionPartFlag: Tensor<Int32> = Tensor(batches.map{ $0.motionFlag.squeezingShape(at: 0) })
            return MotionPart(motion: motionPartTensor, mask: motionPartMask, startFlag: motionPartStartFlag, motionFlag: motionPartFlag)
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
            let sentenceMask = sentence.selfAttentionMask.squeezingShape(at: 1)
            let motionFlag = Tensor<Float>(motionPart.motionFlag).squeezingShape(at: 2)
            let sourceAttentionMask = (sentenceMask * motionFlag.transposed()).expandingShape(at: 0)
            self.init(sentence: sentence, motionPart: motionPart, sourceAttentionMask: sourceAttentionMask)
        }

        public init(copying source: Source, to device: Device) {
            sentence = Sentence(copying: source.sentence, to: device)
            motionPart = MotionPart(copying: source.motionPart, to: device)
            sourceAttentionMask = Tensor(copying: source.sourceAttentionMask, to: device)
        }

        public static func reduceDataBatches(_ batches: [Source]) -> Source {
            let sentence = Sentence.reduceDataBatches(batches.map{ $0.sentence })
            let motionPart = MotionPart.reduceDataBatches(batches.map{ $0.motionPart })
            let sourceAttentionMask: Tensor<Float> = Tensor(batches.map{ $0.sourceAttentionMask.squeezingShape(at: 0) })
            return Source(sentence: sentence, motionPart: motionPart, sourceAttentionMask: sourceAttentionMask)
        }
        
        public func printSource() {
            print("source")
            print("  sentence")
            print("    tokenIds.shape: \(self.sentence.tokenIds.shape)")
            print("    selfAttentionMask.shape: \(self.sentence.selfAttentionMask.shape)")
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

        public static func reduceDataBatches(_ batches: [Target]) -> Target {
            let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.sampleID.squeezingShape(at: 0) })
            let targetMotion: Tensor<Float> = Tensor(batches.map{ $0.motion.squeezingShape(at: 0) })
            let targetStops: Tensor<Float> = Tensor(batches.map{ $0.stops.squeezingShape(at: 0) })
            let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.origMotionFramesCount.squeezingShape(at: 0) })
            return Target(sampleID: sampleID, motion: targetMotion, stops: targetStops, origMotionFramesCount: origMotionFramesCount)
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
        let source = Source(copying: batch.source, to: device)
        let target = Target(copying: batch.target, to: device)
        self.init(source: source, target: target)
    }

    public static func reduceDataBatches(_ batches: [LangMotionBatch]) -> LangMotionBatch {
        let source = Source.reduceDataBatches(batches.map{ $0.source })
        let target = Target.reduceDataBatches(batches.map{ $0.target })
        return LangMotionBatch(source: source, target: target)
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

    public static func neutralMotionFrame() -> Tensor<Float> {
        // one motion frame [1, 47]
        // sampleID: 1, frame: 0, scaled
        return Tensor<Float>([[ -0.612953,  0.0476239,   0.212333,   0.161793,  0.0028466,  0.0190509, -0.0479624, -0.0228922,
                                -0.0694337,   0.387513, -0.0018089,   0.135106,   0.158669,  -0.324085,  -0.165832,    0.35672,
                                -0.0738947,   0.103075,  0.0545089,   0.135437,  -0.221003,  -0.154341,   0.179823,  -0.279305,
                                 0.0844597,  -0.254741,  -0.122429,   0.224831,  0.0616969,   0.324959,  -0.483339,   0.308075,
                                  0.857774, 0.00596291,  0.0430199,   0.373412,  -0.021832,   -0.11818,  -0.083582,  -0.152354,
                                  0.120838,   0.373251, -0.0313812,  -0.261799, -0.0735726, -0.0238353,    1.64312]])
    }
    public static func preprocessTargetMotion(sampleID: Int, motion: Tensor<Float>, maxMotionLength: Int) -> (motionPart: MotionPart, target: Target)
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
        let motionPartMask = Self.makeSelfAttentionDecoderMask(target: motionPartFlag, pad: 0)

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

    public static func subsequentMask(size: Int, shiftRight: Bool = false, on device: Device) -> Tensor<Int32> {
        let attentionShape = [1, size, size]
        let ones = Tensor<Int32>(ones: TensorShape(attentionShape), on: device)
        var mask: Tensor<Int32>
        
        if !shiftRight {
            mask = ones.bandPart(subdiagonalCount: 0, superdiagonalCount: -1)
        } else {
            // https://www.tensorflow.org/tutorials/text/transformer#masking
            mask = 1 - ones.bandPart(subdiagonalCount: -1, superdiagonalCount: 0)
        }
        return mask
    }

    public static func makeSelfAttentionDecoderMask(target: Tensor<Int32>, pad: Int32, on device: Device = Device.defaultTFEager) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target).copy(to: device)
            .replacing(with: Tensor(onesLike: target).copy(to: device), where: target .!= Tensor.init(pad).copy(to: device))
            .expandingShape(at: -2)
        targetMask *= subsequentMask(size: target.shape.last!, shiftRight: false, on: device)
        
        // reverse mask
        targetMask = targetMask.transposed(permutation: [0, 2, 1])

        return Tensor<Float>(targetMask)
    }
}
