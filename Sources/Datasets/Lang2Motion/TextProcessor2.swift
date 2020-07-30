import Foundation
import TensorFlow
import ModelSupport

public struct TextProcessor2 {

    let BOS_WORD = "[CLS]"
    let EOS_WORD = "[SEP]"
    let BLANK_WORD = "[PAD]"
    let UNKNOWN_WORD = "[UNK]"

    public let maxTextSequenceLength: Int
    public let maxMotionLength: Int
    public let vocabulary: Vocabulary
    public let tokenizer: Tokenizer
    public let padId: Int32
    public let bosId: Int32
    public let eosId: Int32
    public let unkId: Int32

    public init(vocabulary: Vocabulary, tokenizer: Tokenizer, maxTextSequenceLength: Int, maxMotionLength: Int) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.maxTextSequenceLength = maxTextSequenceLength
        self.maxMotionLength = maxMotionLength
        self.padId = Int32(self.vocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.vocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.vocabulary.id(forToken: EOS_WORD)!)
        self.unkId = Int32(self.vocabulary.id(forToken: UNKNOWN_WORD)!)
    }

    // TODO: refactor motion out
    public func preprocess(example: Lang2Motion.Example) -> LangMotionBatch {
        let sampleID: Tensor<Int32> = Tensor([Int32(example.sampleID)])

        // source: text
        // ************
        var encodedSource = self.tokenizer.tokenize(example.sentence)
            .prefix(maxTextSequenceLength - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}

        // pad source text
        encodedSource = [bosId] + encodedSource + [eosId]
        let origTokenCount = encodedSource.count
        let paddingCount = encodedSource.count < maxTextSequenceLength ? maxTextSequenceLength - encodedSource.count : 0
        let padding = [Int32](repeating: padId, count: paddingCount)

        encodedSource = encodedSource + padding
        assert(encodedSource.count == maxTextSequenceLength, "encodedSource.count \(encodedSource.count) does not equal maxTextSequenceLength \(maxTextSequenceLength)")

        let tokenIds: Tensor<Int32> = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        let mask: Tensor<Float> = Tensor<Float>(Tensor(zerosLike: tokenIds)
            .replacing(with: Tensor(onesLike: tokenIds), where: tokenIds .!= Tensor.init(padId))
            .expandingShape(at: 1))

        let tokenCount: Tensor<Int32> = Tensor([Int32(origTokenCount)])

        // target: motion
        // **************
        var (motionFrames, motionFlag) = Tensor<Float>(example.motionSample.motion).paddedAndCropped(to: maxMotionLength)
        motionFrames = motionFrames.expandingShape(at: 0)
        motionFlag = motionFlag.expandingShape(at: 0)
        let origMotionFramesCount: Tensor<Int32> = Tensor<Int32>([Int32(example.motionSample.motion.shape[0])])

        let rangeExceptLast = 0..<(motionFrames.shape[1] - 1)
        let targetMotionFrames = motionFrames[0..., rangeExceptLast, 0...]

        motionFlag = motionFlag[0..., rangeExceptLast]
        let targetMask = LangMotionBatch.makeStandardMask(target: motionFlag, pad: 0)

        let targetTruth: Tensor<Float> = motionFrames[0..., 1..., 0...]

        let singleBatch = LangMotionBatch(sampleID: sampleID, 
                tokenIds: tokenIds, mask: mask, tokenCount: tokenCount, 
                targetMotionFrames: targetMotionFrames, targetMask: targetMask,
                targetTruth: targetTruth, origMotionFramesCount: origMotionFramesCount)
        return singleBatch
    }
}
