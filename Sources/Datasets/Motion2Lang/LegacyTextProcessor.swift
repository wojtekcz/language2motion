import Foundation
import TensorFlow
import ModelSupport


// TODO: rename to MotionLangBatchProcessor
public struct LegacyTextProcessor {

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

    /// pads source and target sequences to max sequence length
    public func preprocess(motionSample: MotionSample, maxMotionLength: Int, maxTextSequenceLength: Int) -> MotionLangBatch {

        let motionFrames = Tensor<Float>(motionSample.motion)
        let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))
        
        var (paddedMotion, motionFlag) = motionFrames.paddedAndCropped(to: maxMotionLength)
        paddedMotion = paddedMotion.expandingShape(at: 0)
        motionFlag = motionFlag.expandingShape(at: 0)

        var encodedTarget = self.tokenizer.tokenize(motionSample.annotations[0])
            .prefix(maxTextSequenceLength - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let tPaddingCount = encodedTarget.count < maxTextSequenceLength ? maxTextSequenceLength - encodedTarget.count : 0
        let tPadding = [Int32](repeating: padId, count: tPaddingCount)
        encodedTarget = encodedTarget + tPadding
        assert(encodedTarget.count == maxTextSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxTextSequenceLength \(maxTextSequenceLength)")
        
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        
        let sampleID = Tensor<Int32>(Int32(motionSample.sampleID)).expandingShape(at: 0)

        let singleBatch = MotionLangBatch(sampleID: sampleID, motion: paddedMotion, motionFlag: motionFlag,  origMotionFramesCount: origMotionFramesCount, target: targetTensor, targetPadId: padId)
        return singleBatch
    }

    public func decode(tensor: Tensor<Int32>) -> String {
        let endId = Int32(vocabulary.id(forToken: EOS_WORD)!)
        var words = [String]()
        for scalar in tensor.scalars {
            if let token = vocabulary.token(forId: Int(scalar)) {
                words.append(token)
            }
            if Int(scalar) == endId {
                break
            }
        }
        return words.joined(separator: " ")
    }

}

extension Vocabulary {
    
    public init(fromFile fileURL: URL, specialTokens: [String]) throws {
        let vocabItems = try ( String(contentsOfFile: fileURL.path, encoding: .utf8))
        .components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let dictionary = [String: Int](
                (specialTokens + vocabItems)
                .filter { $0.count > 0 }
                .enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { (v1, v2) in max(v1, v2) })
        self.init(tokensToIds: dictionary )
    }
}
