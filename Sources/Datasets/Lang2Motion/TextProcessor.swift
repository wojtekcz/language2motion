import Foundation
import TensorFlow
import ModelSupport

public struct TextProcessor {

    let BOS_WORD = "[CLS]"
    let EOS_WORD = "[SEP]"
    let BLANK_WORD = "[PAD]"
    let UNKNOWN_WORD = "[UNK]"

    public let vocabulary: Vocabulary
    public let tokenizer: Tokenizer
    public let padId: Int32
    public let bosId: Int32
    public let eosId: Int32
    public let unkId: Int32

    public init(vocabulary: Vocabulary, tokenizer: Tokenizer) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.padId = Int32(self.vocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.vocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.vocabulary.id(forToken: EOS_WORD)!)
        self.unkId = Int32(self.vocabulary.id(forToken: UNKNOWN_WORD)!)
    }

    public func preprocess(sentence: String, maxTextSequenceLength: Int) -> LangMotionBatch.Sentence {
        var encodedSource = self.tokenizer.tokenize(sentence)
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
        let selfAttentionMask: Tensor<Float> = Tensor<Float>(Tensor(zerosLike: tokenIds)
            .replacing(with: Tensor(onesLike: tokenIds), where: tokenIds .!= Tensor.init(padId))
            .expandingShape(at: 1))

        let tokenCount: Tensor<Int32> = Tensor([Int32(origTokenCount)])

        let singleSentence = LangMotionBatch.Sentence(tokenIds: tokenIds, selfAttentionMask: selfAttentionMask, tokenCount: tokenCount)
        return singleSentence
    }
}
