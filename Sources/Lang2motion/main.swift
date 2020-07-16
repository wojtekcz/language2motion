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
let batchSize = 2
let maxSequenceLength =  50
let nEpochs = 1
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
// let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.mini.plist")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.490.plist")
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

var model = LangMotionTransformer(
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

func printBatch(_ batch: LangMotionBatch) {
    print("type: \(type(of:batch))")
    print("sampleID: shape \(batch.sampleID.shape), value \(batch.sampleID)")

    print("source")
    print("  tokenIds.shape: \(batch.tokenIds.shape)")
    print("  mask.shape: \(batch.mask.shape)")
    print("  tokenCount.shape: \(batch.tokenCount.shape), \(batch.tokenCount)")

    print("target")
    print("  targetMotionFrames.shape: \(batch.targetMotionFrames.shape)")
    print("  targetMask.shape: \(batch.targetMask.shape)")
    print("  targetTruth.shape: \(batch.targetTruth.shape)")
    print("  origMotionFramesCount: shape \(batch.origMotionFramesCount.shape), value \(batch.origMotionFramesCount)")
}

/// one example to single batch
print("\nSingle batch")
print("============")
let example = dataset.trainExamples[0]
print("example.sentence: \"\(example.sentence)\"")

let singleBatch = textProcessor.preprocess(example: example)
printBatch(singleBatch)

/// Test model with one batch
/// get a batch
print("\nOne batch:")
print("=========")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch: LangMotionBatch = batches[0]
printBatch(batch)

/// run one batch
print("\nRun one batch:")
print("==============")
let deviceBatch = LangMotionBatch(copying: batch, to: device)
let output = model(deviceBatch)
print("output.shape: \(output.shape)")
