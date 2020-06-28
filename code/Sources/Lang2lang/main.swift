// TODO: instantiate transformer model
// TODO: create one batch (randomly initialized, maybe)
// TODO: run one batch through the model
// TODO: maybe use jupyter notebook

import TensorFlow
import TranslationModels
import Foundation
import ModelSupport
import Datasets


// instantiate model
let sourceVocabSize = 100
let targetVocabSize = 100
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
let batchSize = 1000
let maxSequenceLength =  50

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.balanced.515.csv")

print("\nLoading dataset...")
var dataset = try Lang2Lang(
    datasetURL: dsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { (example: Lang2Lang.Example) -> TranslationBatch in
    // TODO: extract preprocess func from BERT
    // TODO: tokenize
    // TODO: create real TranslationBatch
    let sourceTensor: Tensor<Int32> = Tensor([[1,2,3]])
    let targetTensor: Tensor<Int32> = Tensor([[1,2,3]])
    let sourcePadId: Int32 = 0
    let targetPadId: Int32 = 0
    // let mask: Tensor<Int32> = Tensor([1,2,3])
    // let textBatch = TextBatch(tokenIds: tokenIds, tokenTypeIds: tokenTypeIds, mask: mask)
    // source: Tensor<Int32>, target: Tensor<Int32>, sourcePadId: Int32, targetPadId: Int32
    let singleBatch = TranslationBatch(source: sourceTensor, target: targetTensor, sourcePadId: sourcePadId, targetPadId: targetPadId)
    return singleBatch
        // bertClassifier.bert.preprocess(
        // sequences: [example.text],
        // maxSequenceLength: maxSequenceLength)
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
// print("\nRun one batch:")
// print("==============")
// let classifierOutput = motionClassifier(motionBatch)
// print("classifierOutput.shape: \(classifierOutput.shape)")
