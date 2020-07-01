import Foundation
import TensorFlow
import ModelSupport
import Datasets
import TextModels

let BOS_WORD = "[CLS]"
let EOS_WORD = "[SEP]"
let BLANK_WORD = "[PAD]"
let UNKNOWN_WORD = "[UNK]"


public struct TextProcessor {
    let maxSequenceLength: Int?
    let vocabulary: Vocabulary
    let tokenizer: Tokenizer
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
        
        var encodedSource = self.tokenizer.tokenize("ala ma kota") // FIXME
            .prefix(self.maxSequenceLength! - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        
        encodedSource = [bosId] + encodedSource + [eosId]
        let sPaddingCount = encodedSource.count < maxSequenceLength! ? maxSequenceLength! - encodedSource.count : 0
        let sPadding = [Int32](repeating: padId, count: sPaddingCount)
        encodedSource = encodedSource + sPadding
        assert(encodedSource.count == maxSequenceLength, "encodedSource.count \(encodedSource.count) does not equal maxSequenceLength \(maxSequenceLength!)")

        var encodedTarget = self.tokenizer.tokenize(example.targetSentence)
            .prefix(self.maxSequenceLength! - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let tPaddingCount = encodedTarget.count < maxSequenceLength! ? maxSequenceLength! - encodedTarget.count : 0
        let tPadding = [Int32](repeating: padId, count: tPaddingCount)
        encodedTarget = encodedTarget + tPadding
        assert(encodedTarget.count == maxSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxSequenceLength \(maxSequenceLength!)")
        
        let sourceTensor = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        
        // add padding to target since it will be grouped by sourcelength
        
        // padding is going to be equal to the difference between maxSequence length and the totalEncod
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        let singleBatch = MotionLangBatch(source: sourceTensor, target: targetTensor, sourcePadId: padId, targetPadId: padId)
        
        // print("original source:", example.sourceSentence)
        // print("decoded source:", decode(tensor: singleBatch.tokenIds, vocab: vocabulary))

        // print("max len = \(maxSequenceLength!)")
        // print("encoded target \(encodedTarget.count) last: \(encodedTarget.last!)")
        // print("original target:", example.targetSentence)
        // print("decoded target:", decode(tensor: singleBatch.targetTokenIds, vocab: vocabulary))
        // print("decoded truth:", decode(tensor: singleBatch.targetTruth, vocab: vocabulary))
        return singleBatch
    }
}

func decode(tensor: Tensor<Int32>, vocab: Vocabulary) -> String {
  let endId = Int32(vocab.id(forToken: EOS_WORD)!)
   var words = [String]()
   for scalar in tensor.scalars {
       if Int(scalar) == endId {
           break
       } else
        if let token = vocab.token(forId: Int(scalar)) {
           words.append(token)
       }
   }
   return words.joined(separator: " ")
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
