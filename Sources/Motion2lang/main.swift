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
let runName = "run_18"
let batchSize = 50
let maxMotionLength = 100
let maxTextSequenceLength = 40
let nEpochs = 2

var optimizerOpts = OptimizerOpts(
    peakLearningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    useBiasCorrection: false,
    lrSlopeMultiplier: 2,
    nEpochs: nEpochs
)

//let datasetSize: DatasetSize = .multi_full
let datasetSize: DatasetSize = .mini


print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxMotionLength: \(maxMotionLength)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(optimizerOpts.peakLearningRate)")
print("datasetSize: \(datasetSize)")
// print("stepsPerEpoch: \(optimizerOpts.stepsPerEpoch)")

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
let _ = _ExecutionContext.global

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
var model = MotionLangTransformer(config: config)

/// load model checkpoint
//print("logdirURL: \(logdirURL.path)")
//start_epoch = 50
//let modelName = "model.e\(start_epoch)"
//let modelName = "model.final"
//var model = try! MotionLangTransformer(checkpoint: logdirURL.appendingPathComponent("run_11/checkpoints"), config: config, name: modelName)
// var model = try! MotionLangTransformer(checkpoint: logdirURL.appendingPathComponent("run_17/checkpoints"), config: config, name: "model.e19")

//try! model.writeCheckpoint(to: checkpointURL, name: "model.re-saved2.final")

/// Optimizer
//var optimizer = Adam(for: model, learningRate: learningRate)

optimizerOpts.stepsPerEpoch = dataset.motionSamples.count/batchSize // function of training set size and batching configuration
print("stepsPerEpoch: \(optimizerOpts.stepsPerEpoch)")
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
func greedyDecodeSample(_ sample_id: Int, maxLength: Int = 15, model: MotionLangTransformer, device: Device) {
    let motionSample = dataset.motionSampleDict[sample_id]!
    print("\nsample: \(motionSample.sampleID), \"\(motionSample.annotations[0])\", motion: \(motionSample.timesteps[-1]) sec (\(motionSample.motion.shape[0]) frames)")

    var singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    singleBatch = singleBatch.copy(to: device)
    let decoded = MotionLangDecoder.greedyDecode(model: model, input: singleBatch.source, maxLength: maxLength, startSymbol: textProcessor.bosId, device: device)
    let outputStr = textProcessor.decode(tensor: decoded)
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

// model.move(to: device)
// Context.local.learningPhase = .inference
// for sample in samplesToDecode {
// for _ in 0..<1 {
//    let randomIdx = Int.random(in: 0..<dataset.motionSamples.count)
//    let sampleID = dataset.motionSamples[randomIdx].sampleID
//    greedyDecodeSample(sampleID, maxLength: 40, model: model, device: device)
// }

public func saveCheckpoint<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: MotionLangTransformer) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
    }
}

// public func decodeSamplesAfterEpoch<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: MotionLangTransformer) throws {
//     if event == .epochEnd {
//         Context.local.learningPhase = .inference
//         for sample in samplesToDecode {
//             // TODO: make greedyDecodeSample work on device
//             greedyDecodeSample(sample["sampleID"] as! Int, maxLength: 20, model: model, device: device)
//         }
//     }
// }

// Training loop
print("\nSetting up the training loop")
let trainingProgress = TrainingProgress(metrics: [.loss])
var trainingLoop: TrainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizerWrapper.optimizer,
    lossFunction:  embeddedSoftmaxCrossEntropy,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater, saveCheckpoint]
)

print("\nTraining Transformer for the Motion2lang task!")

try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")

/// Generate motion description
// let sample_id = 733
// greedyDecodeSample(sample_id, maxLength: 20, model: model, device: device)

print("\nFinito.")
