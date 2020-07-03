import Foundation
import TensorFlow
import ModelSupport


public struct TextProcessor {

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

    public init(vocabulary: Vocabulary, tokenizer: Tokenizer, maxSequenceLength: Int? = nil) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.maxSequenceLength = maxSequenceLength
        self.padId = Int32(self.vocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.vocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.vocabulary.id(forToken: EOS_WORD)!)
        self.unkId = Int32(self.vocabulary.id(forToken: UNKNOWN_WORD)!)
    }

    /// pads source and target sequences to max sequence length
    public func preprocess(example: Motion2Lang.Example) -> MotionLangBatch {
        
        let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
        let mfIdx = MotionFrame.cjpMotionFlagIdx
        let motionFlag = Tensor<Int32>(motionFrames[0..., mfIdx...mfIdx].squeezingShape(at: 1))
        let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))

        // var encodedSource = self.tokenizer.tokenize("ala ma kota") // FIXME
        //     .prefix(self.maxSequenceLength! - 2)
        //     .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        
        // encodedSource = [bosId] + encodedSource + [eosId]
        // let sPaddingCount = encodedSource.count < maxSequenceLength! ? maxSequenceLength! - encodedSource.count : 0
        // let sPadding = [Int32](repeating: padId, count: sPaddingCount)
        // encodedSource = encodedSource + sPadding
        // assert(encodedSource.count == maxSequenceLength, "encodedSource.count \(encodedSource.count) does not equal maxSequenceLength \(maxSequenceLength!)")

        var encodedTarget = self.tokenizer.tokenize(example.targetSentence)
            .prefix(self.maxSequenceLength! - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let tPaddingCount = encodedTarget.count < maxSequenceLength! ? maxSequenceLength! - encodedTarget.count : 0
        let tPadding = [Int32](repeating: padId, count: tPaddingCount)
        encodedTarget = encodedTarget + tPadding
        assert(encodedTarget.count == maxSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxSequenceLength \(maxSequenceLength!)")
        
        // let sourceTensor = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        
        // add padding to target since it will be grouped by sourcelength
        
        // padding is going to be equal to the difference between maxSequence length and the totalEncod
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        // let singleBatch = MotionLangBatch(source: sourceTensor, target: targetTensor, sourcePadId: padId, targetPadId: padId)
        //motionFrames: Tensor<Float>, motionFlag: Tensor<Int32>,  origMotionFramesCount: Tensor<Int32>, target: Tensor<Int32>, targetPadId: Int32
        // let motionFrames: Tensor<Float> = Tensor([[1, 2, 3]])
        // let motionFlag: Tensor<Int32> = Tensor([[1, 2, 3]])
        // let origMotionFramesCount: Tensor<Int32> = Tensor([1])
        let singleBatch = MotionLangBatch(motionFrames: motionFrames, motionFlag: motionFlag,  origMotionFramesCount: origMotionFramesCount, target: targetTensor, targetPadId: padId)

        
        // print("original source:", example.sourceSentence)
        // print("decoded source:", decode(tensor: singleBatch.tokenIds, vocab: vocabulary))

        // print("max len = \(maxSequenceLength!)")
        // print("encoded target \(encodedTarget.count) last: \(encodedTarget.last!)")
        // print("original target:", example.targetSentence)
        // print("decoded target:", decode(tensor: singleBatch.targetTokenIds, vocab: vocabulary))
        // print("decoded truth:", decode(tensor: singleBatch.targetTruth, vocab: vocabulary))
        return singleBatch
    }

    public func decode(tensor: Tensor<Int32>) -> String {
        let endId = Int32(vocabulary.id(forToken: EOS_WORD)!)
        var words = [String]()
        for scalar in tensor.scalars {
            if Int(scalar) == endId {
                break
            } else
                if let token = vocabulary.token(forId: Int(scalar)) {
                words.append(token)
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
