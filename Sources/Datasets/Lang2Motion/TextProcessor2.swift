import Foundation
import TensorFlow
import ModelSupport

public struct TextProcessor2 {

    let BOS_WORD = "[CLS]"
    let EOS_WORD = "[SEP]"
    let BLANK_WORD = "[PAD]"
    let UNKNOWN_WORD = "[UNK]"

    public let maxSequenceLength: Int?
    public let vocabulary: Vocabulary
    public let tokenizer: Tokenizer
    public let padId: Int32
    public let bosId: Int32
    public let eosId: Int32
    public let unkId: Int32

    // FIXME: maxSequenceLength shouldn't be optional
    public init(vocabulary: Vocabulary, tokenizer: Tokenizer, maxSequenceLength: Int? = nil) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.maxSequenceLength = maxSequenceLength
        self.padId = Int32(self.vocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.vocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.vocabulary.id(forToken: EOS_WORD)!)
        self.unkId = Int32(self.vocabulary.id(forToken: UNKNOWN_WORD)!)
    }

    public func preprocess(example: Lang2Motion.Example) -> LangMotionBatch {
        // let singleBatch = MotionLangBatch(source: sourceTensor, target: targetTensor, sourcePadId: padId, targetPadId: padId)
        //motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>, target: Tensor<Int32>, targetPadId: Int32

        let sampleID: Tensor<Int32> = Tensor([Int32(example.sampleID)])

        // source: text
        // ************
        var encodedSource = self.tokenizer.tokenize(example.sentence)
            .prefix(maxSequenceLength! - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}

        // pad source text
        encodedSource = [bosId] + encodedSource + [eosId]
        let origTokenCount = encodedSource.count
        let paddingCount = encodedSource.count < maxSequenceLength! ? maxSequenceLength! - encodedSource.count : 0
        let padding = [Int32](repeating: padId, count: paddingCount)

        encodedSource = encodedSource + padding
        assert(encodedSource.count == maxSequenceLength, "encodedSource.count \(encodedSource.count) does not equal maxSequenceLength \(maxSequenceLength!)")

        let tokenIds: Tensor<Int32> = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        let mask: Tensor<Float> = Tensor<Float>(Tensor(zerosLike: tokenIds)
            .replacing(with: Tensor(onesLike: tokenIds), where: tokenIds .!= Tensor.init(padId))
            .expandingShape(at: 1))

        let tokenCount: Tensor<Int32> = Tensor([Int32(origTokenCount)])

        // target: motion
        // **************
        // TODO: stop random cropping
        let maxMotionLength: Int? = 50 // FIXME: move this out
        let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray).paddedOrCropped(to: maxMotionLength!).expandingShape(at: 0)
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(example.motionSample.motionFramesArray.shape[0])])

        let rangeExceptLast = 0..<(motionFrames.shape[1] - 1)
        let targetMotionFrames = motionFrames[0..., rangeExceptLast, 0...]

        let mfIdx = MotionFrame.cjpMotionFlagIdx
        let motionFlag = Tensor<Int32>(targetMotionFrames[0..., 0..., mfIdx...mfIdx]).squeezingShape(at: 2)
        let targetMask = LangMotionBatch.makeStandardMask(target: motionFlag, pad: 0)

        let targetTruth: Tensor<Float> = motionFrames[0..., 1..., 0...]

        let singleBatch = LangMotionBatch(sampleID: sampleID, 
                tokenIds: tokenIds, mask: mask, tokenCount: tokenCount, 
                targetMotionFrames: targetMotionFrames, targetMask: targetMask,
                targetTruth: targetTruth, origMotionFramesCount: origMotionFramesCount)
        return singleBatch
    }
}
