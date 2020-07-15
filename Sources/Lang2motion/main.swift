import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionModels

/// Set training params
let runName = "run_1"
let batchSize = 100
let maxSequenceLength =  50
let nEpochs = 1
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.mini.plist")
let langDatasetURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

/// Select eager or X10 backend
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print(device)

/// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor2(vocabulary: vocabulary, tokenizer: tokenizer, maxSequenceLength: maxSequenceLength)

/// instantiate model
let sourceVocabSize = vocabulary.count
let inputSize = 48 // TODO: get value from dataset
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1

var model = MotionLangTransformer(
    sourceVocabSize: sourceVocabSize, 
    inputSize: inputSize,
    targetVocabSize: targetVocabSize,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

model.move(to: device)

/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    langDatasetURL: langDatasetURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { (example: Lang2Motion.Example) -> LangMotionBatch in    
    let singleBatch = textProcessor.preprocess(example: example)
    return singleBatch
}

print("Dataset acquired.")

/// Test model with one batch
/// get a batch
print("\nOne batch (MotionLangBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch: LangMotionBatch = batches[0]
print("type: \(type(of:batch))")
// print("motionFrames.shape: \(batch.motionFrames.shape)")
// // print("motionFlag.shape: \(batch.motionFlag.shape)")
// print("mask.shape: \(batch.mask.shape)")
// print("origMotionFramesCount.shape: \(batch.origMotionFramesCount.shape)")
// print("origMotionFramesCount: \(batch.origMotionFramesCount)")
// print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")
// print("targetMask.shape: \(batch.targetMask.shape)")
// print("targetTruth.shape: \(batch.targetTruth.shape)")

// /// run one batch
// print("\nRun one batch:")
// print("==============")
// let deviceBatch = MotionLangBatch(copying: batch, to: device)
// let output = model(deviceBatch)
// print("output.shape: \(output.shape)")
