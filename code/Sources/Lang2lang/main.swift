// + instantiate transformer model
// + create one batch (randomly initialized)
// + create real TranslationBatch
// + run one batch through the model

import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets


let maxSequenceLength =  50
let batchSize = 200

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.balanced.515.csv")

// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("uncased_L-12_H-768_A-12/vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let processor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer, maxSequenceLength: maxSequenceLength)

// instantiate model
let sourceVocabSize = vocabulary.count
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1
var model = TransformerModel(
    sourceVocabSize: sourceVocabSize, 
    targetVocabSize: targetVocabSize,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Lang(
    datasetURL: dsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { (example: Lang2Lang.Example) -> TranslationBatch in    
    let singleBatch = processor.preprocess(example: example)
    return singleBatch
}

print("Dataset acquired.")

// get example
print(dataset.trainExamples[0])

// get a batch
print("\nOne batch (TranslationBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch = batches[0]
print("type: \(type(of:batch))")
print("tokenIds.shape: \(batch.tokenIds.shape)")
print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")

print()

// run one batch
print("\nRun one batch:")
print("==============")
let output = model(batch)
print("output.shape: \(output.shape)")
