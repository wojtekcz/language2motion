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
    private let padId: Int32
    private let bosId: Int32
    private let eosId: Int32
    private let unkId: Int32

    public init(vocabulary: Vocabulary, tokenizer: Tokenizer, maxSequenceLength: Int? = nil) {
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.maxSequenceLength = maxSequenceLength
        self.padId = Int32(self.vocabulary.id(forToken: BLANK_WORD)!)
        self.bosId = Int32(self.vocabulary.id(forToken: BOS_WORD)!)
        self.eosId = Int32(self.vocabulary.id(forToken: EOS_WORD)!)
        self.unkId = Int32(self.vocabulary.id(forToken: UNKNOWN_WORD)!)
    }

    public func preprocess(example: Lang2Lang.Example) -> TranslationBatch {
        
        let encodedSource = self.tokenizer.tokenize(example.sourceSentence)
            .prefix(self.maxSequenceLength!)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        
        var encodedTarget = self.tokenizer.tokenize(example.targetSentence)
            .prefix(self.maxSequenceLength! - 2)
            .map{ Int32(self.vocabulary.id(forToken: $0) ?? Int(self.unkId))}
        encodedTarget = [bosId] + encodedTarget + [eosId]
        let paddingCount = encodedTarget.count < maxSequenceLength! ? maxSequenceLength! - encodedTarget.count : 0
        let padding = [Int32](repeating: padId, count: paddingCount)
        encodedTarget = encodedTarget + padding
        assert(encodedTarget.count == maxSequenceLength, "encodedTarget.count \(encodedTarget.count) does not equal maxSequenceLength \(maxSequenceLength!)")
        
        let sourceTensor = Tensor<Int32>.init(encodedSource).expandingShape(at: 0)
        
        // add padding to target since it will be grouped by sourcelength
        
        // padding is going to be equal to the difference between maxSequence length and the totalEncod
        let targetTensor = Tensor<Int32>.init( encodedTarget).expandingShape(at: 0)
        let singleBatch = TranslationBatch(source: sourceTensor, target: targetTensor, sourcePadId: padId, targetPadId: padId)
        
        print("original source:", example.sourceSentence)
        print("decoded source:", decode(tensor: singleBatch.tokenIds, vocab: vocabulary))

        print("max len = \(maxSequenceLength!)")
        print("encoded target \(encodedTarget.count) last: \(encodedTarget.last!)")
        print("original target:", example.targetSentence)
        print("decoded target:", decode(tensor: singleBatch.targetTokenIds, vocab: vocabulary))
        print("decoded truth:", decode(tensor: singleBatch.targetTruth, vocab: vocabulary))
        return singleBatch
    }

    /// Preprocesses an array of text sequences and prepares them for processing with BERT.
    /// Preprocessing mainly consists of tokenization.
    ///
    /// - Parameters:
    ///   - sequences: Text sequences (not tokenized).
    ///   - maxSequenceLength: Maximum sequence length supported by the text perception module.
    ///     This is mainly used for padding the preprocessed sequences. If not provided, it
    ///     defaults to this model's maximum supported sequence length.
    ///   - tokenizer: Tokenizer to use while preprocessing.
    ///
    /// - Returns: Text batch that can be processed by BERT.
    // public func preprocess(sequences: [String], maxSequenceLength: Int? = nil) -> Tensor<Int32> {
    //     let maxSequenceLength = maxSequenceLength ?? self.maxSequenceLength
    //     var sequences = sequences.map(tokenizer.tokenize)

    //     // Truncate the sequences based on the maximum allowed sequence length, while accounting
    //     // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
    //     // simple heuristic which will truncate the longer sequence one token at a time. This makes 
    //     // more sense than truncating an equal percent of tokens from each sequence, since if one
    //     // sequence is very short then each token that is truncated likely contains more
    //     // information than respective tokens in longer sequences.
    //     var totalLength = sequences.map { $0.count }.reduce(0, +)
    //     let totalLengthLimit = maxSequenceLength! - 1 - sequences.count
    //     while totalLength >= totalLengthLimit {
    //         let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count })!.0
    //         sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
    //         totalLength = sequences.map { $0.count }.reduce(0, +)
    //     }

    //     // The convention in BERT is:
    //     //   (a) For sequence pairs:
    //     //       tokens:       [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    //     //       tokenTypeIds: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    //     //   (b) For single sequences:
    //     //       tokens:       [CLS] the dog is hairy . [SEP]
    //     //       tokenTypeIds: 0     0   0   0  0     0 0
    //     // where "tokenTypeIds" are used to indicate whether this is the first sequence or the
    //     // second sequence. The embedding vectors for `tokenTypeId = 0` and `tokenTypeId = 1` were
    //     // learned during pre-training and are added to the WordPiece embedding vector (and
    //     // position vector). This is not *strictly* necessary since the [SEP] token unambiguously
    //     // separates the sequences. However, it makes it easier for the model to learn the concept
    //     // of sequences.
    //     //
    //     // For classification tasks, the first vector (corresponding to `[CLS]`) is used as the
    //     // "sentence embedding". Note that this only makes sense because the entire model is
    //     // fine-tuned under this assumption.
    //     var tokens = ["[CLS]"]
    //     for (sequenceId, sequence) in sequences.enumerated() {
    //         for token in sequence {
    //             tokens.append(token)
    //         }
    //         tokens.append("[SEP]")
    //     }
    //     let tokenIds = tokens.map { Int32(vocabulary.id(forToken: $0)!) }

    //     // The mask is set to `true` for real tokens and `false` for padding tokens. This is so
    //     // that only real tokens are attended to.
    //     let mask = [Int32](repeating: 1, count: tokenIds.count)

    //     return TextBatch(
    //         tokenIds: Tensor(tokenIds).expandingShape(at: 0),
    //         tokenTypeIds: Tensor(tokenTypeIds).expandingShape(at: 0),
    //         mask: Tensor(mask).expandingShape(at: 0))
    // }
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
