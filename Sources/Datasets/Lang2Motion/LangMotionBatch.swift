import Foundation
import TensorFlow
import ModelSupport

public typealias LangMotionBatch = LabeledData<LangMotion.Source, LangMotion.Target>

public struct LangMotion {

    public struct Sentence {
        public var tokenIds: Tensor<Int32>          // bs x maxTextSequenceLength
        public var selfAttentionMask: Tensor<Float> // bs x maxMotionLength x maxTextSequenceLength // or x 1?
        public let tokenCount: Tensor<Int32>        // bs

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

    public struct MotionPart {
        public var motion: Tensor<Float>               // bs x maxMotionLength x nbJoints
        public var discreteMotion: Tensor<Int32>       // bs x maxMotionLength x nbJoints // mask value needed? padding needed?

        public var decSelfAttentionMask: Tensor<Float> // bs x maxMotionLength x maxMotionLength
        public var motionFlag: Tensor<Int32>           // bs x maxMotionLength x 1
        public var segmentIDs: Tensor<Int32>           // bs x maxMotionLength x 1

        public init(motion: Tensor<Float>, discreteMotion: Tensor<Int32>, decSelfAttentionMask: Tensor<Float>, motionFlag: Tensor<Int32>, segmentIDs: Tensor<Int32>) {
            self.motion = motion
            self.discreteMotion = discreteMotion
            
            self.decSelfAttentionMask = decSelfAttentionMask
            self.motionFlag = motionFlag
            self.segmentIDs = segmentIDs

            assert(motionFlag.shape.count == 3 && motionFlag.shape[2]==1)
            assert(segmentIDs.shape.count == 3 && segmentIDs.shape[2]==1)
        }

        public init(copying motionPart: MotionPart, to device: Device) {
            motion = Tensor<Float>(copying: motionPart.motion, to: device)
            discreteMotion = Tensor<Int32>(copying: motionPart.discreteMotion, to: device)

            decSelfAttentionMask = Tensor<Float>(copying: motionPart.decSelfAttentionMask, to: device)
            motionFlag = Tensor<Int32>(copying: motionPart.motionFlag, to: device)
            segmentIDs = Tensor<Int32>(copying: motionPart.segmentIDs, to: device)
        }

        public static func reduceDataBatches(_ batches: [MotionPart]) -> MotionPart {
            let motionPartTensor: Tensor<Float> = Tensor(batches.map{ $0.motion.squeezingShape(at: 0) })
            let discreteMotionPartTensor: Tensor<Int32> = Tensor(batches.map{ $0.discreteMotion.squeezingShape(at: 0) })
            let motionPartMask: Tensor<Float> = Tensor(batches.map{ $0.decSelfAttentionMask.squeezingShape(at: 0) })
            let motionPartFlag: Tensor<Int32> = Tensor(batches.map{ $0.motionFlag.squeezingShape(at: 0) })
            let segmentIDs: Tensor<Int32> = Tensor(batches.map{ $0.segmentIDs.squeezingShape(at: 0) })
            return MotionPart(motion: motionPartTensor, discreteMotion: discreteMotionPartTensor, decSelfAttentionMask: motionPartMask, motionFlag: motionPartFlag, segmentIDs: segmentIDs)
        }

        public func printMotionPart() {
            print("motionPart")
            print("  motion.shape: \(self.motion.shape)")
            print("  discreteMotion.shape: \(self.discreteMotion.shape)")
            print("  decSelfAttentionMask.shape: \(self.decSelfAttentionMask.shape)")
            print("  motionFlag.shape: \(self.motionFlag.shape)")
            print("  segmentIDs.shape: \(self.segmentIDs.shape)")
        }
    }

    public struct Source {
        public var sentence: Sentence
        public var motionPart: MotionPart
        public var sourceAttentionMask: Tensor<Float> // bs x maxMotionLength x maxTextSequenceLength

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
            print("    decSelfAttentionMask.shape: \(self.motionPart.decSelfAttentionMask.shape)")
            print("    motionFlag.shape: \(self.motionPart.motionFlag.shape)")
            print("    segmentIDs.shape: \(self.motionPart.segmentIDs.shape)")
            print("  sourceAttentionMask.shape: \(self.sourceAttentionMask.shape)")
        }
    }

    public struct Target {
        public let sampleID: Tensor<Int32>        // bs

        public var motion: Tensor<Float>          // bs x maxMotionLength x nbJoints
        public var discreteMotion: Tensor<Int32>  // bs x maxMotionLength x nbJoints // mask value needed? padding needed?
        public var stops: Tensor<Float>           // bs x maxMotionLength
        public var segmentIDs: Tensor<Int32>      // bs x maxMotionLength

        public let origMotionFramesCount: Tensor<Int32> // bs

        public init(sampleID: Tensor<Int32>, motion: Tensor<Float>, discreteMotion: Tensor<Int32>, stops: Tensor<Float>, segmentIDs: Tensor<Int32>, origMotionFramesCount: Tensor<Int32>) {
            self.sampleID = sampleID

            self.motion = motion
            self.discreteMotion = discreteMotion
            self.stops = stops
            self.segmentIDs = segmentIDs
            self.origMotionFramesCount = origMotionFramesCount
        }

        public init(copying target: Target, to device: Device) {
            sampleID = Tensor<Int32>(copying: target.sampleID, to: device)

            motion = Tensor<Float>(copying: target.motion, to: device)
            discreteMotion = Tensor<Int32>(copying: target.discreteMotion, to: device)
            stops = Tensor<Float>(copying: target.stops, to: device)
            segmentIDs = Tensor<Int32>(copying: target.segmentIDs, to: device)
            origMotionFramesCount = Tensor<Int32>(copying: target.origMotionFramesCount, to: device)
        }

        public static func reduceDataBatches(_ batches: [Target]) -> Target {
            let sampleID: Tensor<Int32> = Tensor(batches.map{ $0.sampleID.squeezingShape(at: 0) })
            let motion: Tensor<Float> = Tensor(batches.map{ $0.motion.squeezingShape(at: 0) })
            let discreteMotion: Tensor<Int32> = Tensor(batches.map{ $0.discreteMotion.squeezingShape(at: 0) })
            let stops: Tensor<Float> = Tensor(batches.map{ $0.stops.squeezingShape(at: 0) })
            let segmentIDs: Tensor<Int32> = Tensor(batches.map{ $0.segmentIDs.squeezingShape(at: 0) })
            let origMotionFramesCount: Tensor<Int32> = Tensor(batches.map{ $0.origMotionFramesCount.squeezingShape(at: 0) })
            return Target(sampleID: sampleID, motion: motion, discreteMotion: discreteMotion, stops: stops, segmentIDs: segmentIDs, origMotionFramesCount: origMotionFramesCount)
        }

        public func printTarget() {
            print("target")
            print("  sampleID: (shape: \(self.sampleID.shape), value: \(self.sampleID))")

            print("  motion.shape: \(self.motion.shape)")
            print("  discreteMotion.shape: \(self.discreteMotion.shape)")
            print("  stops.shape: \(self.stops.shape)")
            print("  segmentIDs.shape: \(self.segmentIDs.shape)")
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

    public enum MotionSegment: Int32, CaseIterable {
        case padding = 0
        case motion = 1
        case start = 2
        case stop = 3
    }

    public static func padAndSegmentMotion(_ inputMotion: Tensor<Float>, to width: Int) -> (motion: Tensor<Float>, motionFlag: Tensor<Int32>, segmentIDs: Tensor<Int32>) {
        // pads two-dimensional tensor
        assert(inputMotion.shape.count == 2)
        let currentWidth = inputMotion.shape[0]
        let paddingSize = Swift.max(width - currentWidth, 0)
        let sizes: [(before: Int, after: Int)] = [(before: 0, after: paddingSize), (before: 0, after: 0)]
        
        let motion = inputMotion[0..<width].padded(forSizes: sizes)
        
        var motionFlag = Tensor<Float>(repeating: 1, shape: [currentWidth])
        motionFlag = motionFlag[0..<width].padded(forSizes: [(before: 0, after: paddingSize)], with: 0)
        motionFlag[currentWidth-1] = Tensor(0) // STOP frame

        // form segment ids
        var segmentIDs = Tensor<Float>(repeating: Float(MotionSegment.motion.rawValue), shape: [currentWidth]) // MOTION
        segmentIDs[0] = Tensor(Float(MotionSegment.start.rawValue)) // START
        segmentIDs = segmentIDs[0..<width].padded(forSizes: [(before: 0, after: paddingSize)], with: Float(MotionSegment.padding.rawValue)) // PADDING
        segmentIDs[currentWidth-1] = Tensor(Float(MotionSegment.stop.rawValue)) // STOP

        return (motion: motion, motionFlag: Tensor<Int32>(motionFlag), segmentIDs: Tensor<Int32>(segmentIDs)) // all 1-dim tensors
    }

    public static func preprocessTargetMotion(sampleID: Int, motion: Tensor<Float>, maxMotionLength: Int, discretizer: MotionDiscretizer) -> (motionPart: MotionPart, target: Target)
    {
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(motion.shape[0])])
        
        let neutralMotionFrame = Self.neutralMotionFrame()
        
        // add start and stop "neutral" frames
        let motion2 = Tensor(concatenating: [neutralMotionFrame, motion[0..<maxMotionLength-1], neutralMotionFrame], alongAxis: 0)

        let (paddedMotion, motionFlag, segmentIDs) = padAndSegmentMotion(motion2, to: maxMotionLength+1)
        
        // source (motionPart & motion flag)
        let rangeExceptLast = 0..<(paddedMotion.shape[0] - 1)
        let motionPartTensor = paddedMotion[rangeExceptLast, 0...]
        let discreteMotionPartTensor = discretizer.transform(motionPartTensor) //Tensor<Int32>([[0]])

        let motionPartFlag = motionFlag[rangeExceptLast]
        let motionPartSegmentIDs = segmentIDs[rangeExceptLast]
        let motionPartMask = Self.makeSelfAttentionDecoderMask(target: motionPartFlag, pad: 0)

        let motionPart = MotionPart(
            motion: motionPartTensor.expandingShape(at: 0),
            discreteMotion: discreteMotionPartTensor.expandingShape(at: 0),
            decSelfAttentionMask: motionPartMask,
            motionFlag: motionPartFlag.expandingShape(at: [0, 2]),
            segmentIDs: motionPartSegmentIDs.expandingShape(at: [0, 2])
        )

        // target (motion & stops)
        let targetMotion: Tensor<Float> = paddedMotion[1..., 0...]
        let targetDiscreteMotion: Tensor<Int32> = discretizer.transform(targetMotion)
        let targetMotionFlag = motionFlag[1...].expandingShape(at: 0)
        let targetStops: Tensor<Float> = 1.0 - Tensor<Float>(targetMotionFlag)
        let targetSegmentIDs = segmentIDs[1...].expandingShape(at: 0)

        
        let target = Target(sampleID: Tensor([Int32(sampleID)]), motion: targetMotion.expandingShape(at: 0), discreteMotion: targetDiscreteMotion.expandingShape(at: 0), stops: targetStops, segmentIDs: targetSegmentIDs, origMotionFramesCount: origMotionFramesCount)
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
