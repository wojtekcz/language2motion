import Foundation
import TensorFlow
import ModelSupport

public typealias MotionLangBatch = LabeledData<MotionLang.Source, MotionLang.Target>

public struct MotionLang {
    public struct Source {
        // sampleID
        public let sampleID: Tensor<Int32>

        // motion
        public var motion: Tensor<Float>
        // motionFlag
        // ...
        
        // mask // encoder self-attention mask
        public var mask: Tensor<Float>

        // origMotionLength
        public let origMotionFramesCount: Tensor<Int32>
        
        // targetTokenIds, sentenceTokenIds
        public var targetTokenIds: Tensor<Int32>
        // targetMask // decoder self-attention mask?
        public var targetMask: Tensor<Float>
        // sourceAttentionMask // decoder source attention mask?
        // origSentenceLength

        public init(
            sampleID: Tensor<Int32>,
            motion: Tensor<Float>,
            mask: Tensor<Float>,
            origMotionFramesCount: Tensor<Int32>,
            
            targetTokenIds: Tensor<Int32>,
            targetMask: Tensor<Float>
        ) {
            self.sampleID = sampleID
            self.motion = motion
            self.mask = mask
            self.origMotionFramesCount = origMotionFramesCount

            self.targetTokenIds = targetTokenIds
            self.targetMask = targetMask
        }
        
        public init(copying source: Source, to device: Device) {
            sampleID = Tensor(copying: source.sampleID, to: device)
            motion = Tensor(copying: source.motion, to: device)
            mask = Tensor(copying: source.mask, to: device)
            origMotionFramesCount = Tensor(copying: source.origMotionFramesCount, to: device)

            targetTokenIds = Tensor(copying: source.targetTokenIds, to: device)
            targetMask = Tensor(copying: source.targetMask, to: device)
        }
        
        public static func reduceDataBatches(_ batches: [Source]) -> Source {
            let sampleID = Tensor(batches.map{ $0.sampleID.squeezingShape(at: 0) })

            let motion = Tensor(batches.map{ $0.motion.squeezingShape(at: 0) })
            let mask = Tensor(batches.map{ $0.mask.squeezingShape(at: 0) })
            let origMotionFramesCount = Tensor(batches.map{$0.origMotionFramesCount})

            let targetTokenIds = Tensor(batches.map{ $0.targetTokenIds.squeezingShape(at: 0) })
            let targetMask = Tensor(batches.map{ $0.targetMask.squeezingShape(at: 0) })
            return Source(
                sampleID: sampleID,
                motion: motion,
                mask: mask,
                origMotionFramesCount: origMotionFramesCount,

                targetTokenIds: targetTokenIds,
                targetMask: targetMask
            )
        }
    }
    public struct Target {
        // sampleID
        // targetTruthTokenIds
        public var targetTruth: Tensor<Int32>

        public init(targetTruth: Tensor<Int32>) {
            self.targetTruth = targetTruth
        }
        
        public init(copying target: Target, to device: Device) {
            targetTruth = Tensor(copying: target.targetTruth, to: device)
        }
        
        public static func reduceDataBatches(_ batches: [Target]) -> Target {
            let targetTruth = Tensor(batches.map{ $0.targetTruth.squeezingShape(at: 0) })
            return Target(targetTruth: targetTruth)
        }
    }
}

extension MotionLangBatch {
    public typealias MLSource = MotionLang.Source
    public typealias MLTarget = MotionLang.Target

    public var source: MLSource { get { return data } }
    public var target: MLTarget { get { return label } }

    public init(source: MLSource, target: MLTarget) {
        self.init(data: source, label: target)
    }

    public init(copying batch: MotionLangBatch, to device: Device) {
        let data = MLSource(copying: batch.data, to: device)
        let label = MLTarget(copying: batch.label, to: device)
        self.init(data: data, label: label)
    }
    
    public static func reduceDataBatches(_ batches: [MotionLangBatch]) -> MotionLangBatch {
        let source = MLSource.reduceDataBatches(batches.map{ $0.source })
        let target = MLTarget.reduceDataBatches(batches.map{ $0.target })
        return MotionLangBatch(source: source, target: target)
    }
    
    public func copy(to device: Device) -> Self {
        return Self(copying: self, to: device)
    }

    public init(sampleID: Tensor<Int32>, motion: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>, target: Tensor<Int32>, targetPadId: Int32) {
        
        let mask = Tensor<Float>(Tensor(zerosLike: motionFlag)
            .replacing(with: Tensor(onesLike: motionFlag), where: motionFlag .!= Tensor.init(0))
            .expandingShape(at: 1))

        let rangeExceptLast = 0..<(target.shape[1] - 1)
        let targetTokenIds = target[0...,rangeExceptLast]
        let targetTruth = target[0..., 1...]
//        self.targetMask = MotionLangBatch.makeStandardMask(target: self.targetTokenIds, pad: targetPadId)

        var motionPartMask = Self.makeStandardMask(target: targetTokenIds, pad: targetPadId, shiftRight: true, on: Device.defaultTFEager)
        let motionLen = Int(targetTokenIds.sum().scalar!)
        motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
        motionPartMask = abs(motionPartMask)
        
        let source = MLSource(sampleID: sampleID, motion: motion, mask: mask, origMotionFramesCount: origMotionFramesCount, targetTokenIds: targetTokenIds, targetMask: motionPartMask)

        let target = MLTarget(targetTruth: targetTruth)
        self.init(source: source, target: target)
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

    public static func makeStandardMask(target: Tensor<Int32>, pad: Int32, shiftRight: Bool = false, on device: Device) -> Tensor<Float> {
        var targetMask = Tensor(zerosLike: target).copy(to: device)
            .replacing(with: Tensor(onesLike: target, on: device), where: target .!= Tensor.init(pad))
            .expandingShape(at: -2)
        targetMask *= subsequentMask(size: target.shape.last!, shiftRight: shiftRight, on: device)
        return Tensor<Float>(targetMask)
    }
}

// TODO: add targetTokenCount and sampleID
