import Foundation
import TensorFlow
import Datasets
import MotionModels
import ImageClassificationModels
import TextModels
import ModelSupport


let batchSize = 2
let tensorWidth = 60

print("batchSize: \(batchSize)")
print("tensorWidth: \(tensorWidth)")

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.normalized.100.plist")
let labelsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

print("\nLoading dataset...")
let dataset = try! Motion2Label2(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: tensorWidth,
    batchSize: batchSize
) { 
    // TODO: move this to the dataset
    (example: Motion2LabelExample) -> LabeledMotionBatch in
    let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
    let motionFlag = Tensor<Int32>(motionFrames[0..., 44...44])
    let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag)
    let label = Tensor<Int32>(Int32(example.label!.idx))
    return LabeledMotionBatch(
        data: motionBatch, 
        label: label
    )
}

print("dataset.trainingExamples.count: \(dataset.trainingExamples.count)")
print("dataset.validationExamples.count: \(dataset.validationExamples.count)")

// print("dataset.trainingExamples[0]: \(dataset.trainingExamples[0])")

// for (epoch, epochBatches) in dataset.trainingEpochs.prefix(5).enumerated() {
//     print("[Epoch \(epoch + 1)]")
//     for _ in epochBatches {
//         // print(7)
//     }
// }

// print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
// for _ in dataset.validationBatches {
//     // print(8)
// }

// instantiate model
var hiddenSize = 768
let classCount = 5
var featureExtractor = ResNet(classCount: hiddenSize, depth: .resNet18, downsamplingInFirstStage: false, channelCount: 1)

// instantiate BERT
var caseSensitive: Bool = false
var subDirectory: String = "uncased_L-12_H-768_A-12"
let directory = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let vocabularyURL = directory
    .appendingPathComponent(subDirectory)
    .appendingPathComponent("vocab.txt")

let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary,
    caseSensitive: caseSensitive, unknownToken: "[UNK]", maxTokenLength: nil)

var variant: BERT.Variant = .bert          
// var hiddenSize: Int = 768
var hiddenLayerCount: Int = 12
var attentionHeadCount: Int = 12
var intermediateSize: Int = hiddenSize*4 // 3072/768=4

var transformerEncoder = BERT(
    variant: variant,
    vocabulary: vocabulary,
    tokenizer: tokenizer,
    caseSensitive: caseSensitive,
    hiddenSize: hiddenSize,
    hiddenLayerCount: hiddenLayerCount,
    attentionHeadCount: attentionHeadCount,
    intermediateSize: intermediateSize,
    intermediateActivation: gelu,
    hiddenDropoutProbability: 0.1,
    attentionDropoutProbability: 0.1,
    maxSequenceLength: 512,
    typeVocabularySize: 2,
    initializerStandardDeviation: 0.02,
    useOneHotEmbeddings: false)

var motionClassifier = MotionClassifier(featureExtractor: featureExtractor, transformerEncoder: transformerEncoder, classCount: classCount)


// get a batch
print("\nOne batch (LabeledMotionBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch = batches[0]
print("type: \(type(of:batch))")
print("\nOne motionBatch")
let motionBatch = batch.data
print("type: \(type(of:motionBatch))")
print("motionFrames.shape: \(motionBatch.motionFrames.shape)")
print("motionFlag.shape: \(motionBatch.motionFlag.shape)")


// run one batch
print("\nRun one batch:")
print("==============")
let classifierOutput = motionClassifier(motionBatch)
print("classifierOutput.shape: \(classifierOutput.shape)")

print("\nFinito.")
