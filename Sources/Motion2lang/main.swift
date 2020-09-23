import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionLangModels
import TrainingLoop
import x10_optimizers_optimizer

/// Set training params
let runName = "run_10"
let batchSize = 100
//let batchSize = 300
let maxMotionLength = 50
let maxTextSequenceLength = 40
let nEpochs = 10

var optimizerOpts = OptimizerOpts(
    peakLearningRate: 2e-5,
    beta1: 0.9,
    beta2: 0.999,
    useBiasCorrection: false,
    lrSlopeMultiplier: 2,
    nEpochs: nEpochs
)

//let datasetSize: DatasetSize = .multi_full
let datasetSize: DatasetSize = .multi_midi


print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxMotionLength: \(maxMotionLength)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(optimizerOpts.peakLearningRate)")
print("datasetSize: \(datasetSize)")
print("stepsPerEpoch: \(optimizerOpts.stepsPerEpoch)")

#if os(macOS)
//    let dataURL = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Motion2lang/", isDirectory: true)
let rundirURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
let checkpointURL = rundirURL.appendingPathComponent("checkpoints", isDirectory: true)

// FIXME: how to make macOS builds use filesystem in read/write mode?
#if os(Linux)
    try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
#endif

/// Select eager or X10 backend

// let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")

/// X10 warm-up
let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
let eagerTensorSum = eagerTensor1 + eagerTensor2
//print(eagerTensorSum)
//print(eagerTensor1.device)
let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
//print(x10Tensor2.device)

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
// let _ = _ExecutionContext.global

/// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = LegacyTextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)


/// load dataset
print("\nLoading dataset...")

var dataset = try Motion2Lang(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 20,
    maxMotionLength: 100,
    trainTestSplit: 0.9,
    device: device
) { (motionSample: MotionSample) -> MotionLangBatch in
    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    return singleBatch
}

print("Dataset acquired.")

print(dataset.motionSamples.count)

optimizerOpts.stepsPerEpoch = dataset.motionSamples.count/batchSize // function of training set size and batching configuration

/// instantiate model
print("instantiate model")
let modelSize = 128
let config = MotionLangTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47,
    layerCount: 6,
    modelSize: modelSize,
    feedForwardSize: 512,
    headCount: 4,
    dropoutProbability: 0.1,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500
)

var start_epoch = 0

/// create new model
//var model = MotionLangTransformer(config: config)

/// load model checkpoint
//print("logdirURL: \(logdirURL.path)")
start_epoch = 17
let modeName = "model.e\(start_epoch)"
//let modeName = "model.final"
var model = try! MotionLangTransformer(checkpoint: logdirURL.appendingPathComponent("run_9/checkpoints"), config: config, name: modeName)


/// Optimizer
//var optimizer = Adam(for: model, learningRate: learningRate)

let optimizerWrapper = OptimizerWrapper(opts: optimizerOpts, model: model)

/// stats recorder
let statsRecorder = StatsRecorder(logdirURL: rundirURL)

@differentiable(wrt: y_pred)
func embeddedSoftmaxCrossEntropy(y_pred: Tensor<Float>, y_true: MotionLangBatch.MLTarget) -> Tensor<Float> {
    let resultSize = y_true.targetTruth.shape.last! * y_true.targetTruth.shape.first!
    let logits = y_pred.reshaped(to: [resultSize, -1])
    let labels = y_true.targetTruth.reshaped(to: [-1])
    // TODO: ignore padded entries
    return softmaxCrossEntropy(logits: logits, labels: labels)
}

/// Set up decoding
func greedyDecodeSample(_ sample_id: Int, maxLength: Int = 15) {
    let motionSample = dataset.motionSampleDict[sample_id]!
    print("\nsample: \(motionSample.sampleID), \"\(motionSample.annotations[0])\", motion: \(motionSample.timesteps[-1]) sec (\(motionSample.motion.shape[0]) frames)")

    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    let out = MotionLangDecoder.greedyDecode(model: model, input: singleBatch.source, maxLength: maxLength, startSymbol: textProcessor.bosId)
    let outputStr = textProcessor.decode(tensor: out)
    print("decoded: \"\(outputStr)\"")
}

let samplesToDecode = [
    ["sampleID": dataset.motionSamples[0].sampleID, "text": dataset.motionSamples[0].annotations[0]], // for small dataset
//    ["sampleID": 733, "text": "Ala ma kota."], // for .micro dataset
//    ["sampleID": 1242, "text": "Ala ma kota."], // for .multi_mini dataset
//    ["sampleID": 449, "text": "A person runs forward."],
//    ["sampleID": 3921, "text": "A human is swimming."],
//    ["sampleID": 843, "text": "A person walks."],
//    ["sampleID": 1426, "text": "A person plays the air guitar."],
//    ["sampleID": 1292, "text": "A person performs a squat."],
//    ["sampleID": 1315, "text": "A human raises their left foot and touches it with the right hand."]
]

// Training loop
print("\nSetting up the training loop")
let trainingProgress = TrainingProgress(metrics: [.loss])
var trainingLoop: TrainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizerWrapper.optimizer,
    lossFunction:  embeddedSoftmaxCrossEntropy,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater]
)

print("\nTraining Transformer for the Motion2lang task!")
// FIXME: epoch loop workaround for checkpoint saving
for epochIndex in start_epoch..<start_epoch+nEpochs {
    print("epoch \(epochIndex+1)/\(start_epoch + nEpochs)")
    statsRecorder.epochIndex = epochIndex
    try! trainingLoop.fit(&model, epochs: 1, on: device)
    try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")

    Context.local.learningPhase = .inference
    model.move(to: Device.defaultTFEager)
    for sample in samplesToDecode {
        greedyDecodeSample(sample["sampleID"] as! Int, maxLength: 20)
    }
    model.move(to: device)
}

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")

/// Generate motion description
let sample_id = 733
greedyDecodeSample(sample_id, maxLength: 20)

print("\nFinito.")
