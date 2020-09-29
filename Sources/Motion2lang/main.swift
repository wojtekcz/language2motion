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
let runName = "run_31"
let batchSize = 150
let maxMotionLength = 150
let maxTextSequenceLength = 50
let nEpochs = 50

var optimizerOpts = OptimizerOpts(
    peakLearningRate: 5e-4,
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

#if os(macOS)
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

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
let _ = _ExecutionContext.global

/// Select eager or X10 backend
let device = Device.defaultXLA
// let device = Device.defaultTFEager
print("backend: \(device)")

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
    maxMotionLength: 150,
    trainTestSplit: 0.9,
    device: device
) { (motionSample: MotionSample) -> MotionLangBatch in
    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    return singleBatch
}

print("Dataset acquired.")

/// instantiate model
print("instantiate model")
let config = MotionLangTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47,
    layerCount: 6,
    encoderDepth: 256,
    decoderDepth: 256,
    feedForwardSize: 2048,
    headCount: 16,
    dropoutProbability: 0.1,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500
)

/// create new model
var model = MotionLangTransformer(config: config)

/// load model checkpoint
// var model = try! MotionLangTransformer(checkpoint: logdirURL.appendingPathComponent("run_24/checkpoints"), config: config, name: "model.e5")

@differentiable(wrt: y_pred)
func embeddedSoftmaxCrossEntropy(y_pred: Tensor<Float>, y_true: MotionLangBatch.MLTarget) -> Tensor<Float> {
    let resultSize = y_true.targetTruth.shape.last! * y_true.targetTruth.shape.first!
    let logits = y_pred.reshaped(to: [resultSize, -1])
    let labels = y_true.targetTruth.reshaped(to: [-1])

    // masking padded entries
    @differentiable
    func _none(t: Tensor<Float>) -> Tensor<Float> { t }
    let sceLosses = softmaxCrossEntropy(logits: logits, labels: labels, reduction: _none)
    let lossesMaskInt32 = labels.replacing(with: Tensor<Int32>(repeating: 1, shape: labels.shape, on: device), where: labels .!= Tensor<Int32>(0, on: device))
    let lossesMask = Tensor<Float>(lossesMaskInt32)
    let nonPaddedCount = Float(lossesMask.sum().scalar!)
    return (sceLosses * lossesMask).sum()/nonPaddedCount
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

//let samplesToDecode: [[String:Any]] = [
//    ["sampleID": dataset.motionSamples[0].sampleID, "text": dataset.motionSamples[0].annotations[0]], // for small dataset
//    ["sampleID": 733, "text": "Ala ma kota."], // for .micro dataset
//    ["sampleID": 1242, "text": "Ala ma kota."], // for .multi_mini dataset
//    ["sampleID": 449, "text": "A person runs forward."],
//    ["sampleID": 3921, "text": "A human is swimming."],
//    ["sampleID": 843, "text": "A person walks."],
//    ["sampleID": 1426, "text": "A person plays the air guitar."],
//    ["sampleID": 1292, "text": "A person performs a squat."],
//    ["sampleID": 1315, "text": "A human raises their left foot and touches it with the right hand."]
//]

let nSamples = 5
let samplesToDecode: [[String: Any]] = (0..<nSamples).map { (i) -> [String: Any] in
    let randomIdx = Int.random(in: 0..<dataset.motionSamples.count)
    let ms = dataset.motionSamples[randomIdx]
    return ["sampleID": ms.sampleID, "text": ms.annotations[0]]
}

public func saveCheckpoint<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: MotionLangTransformer) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
    }
}

public func decodeSamplesAfterEpoch<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: MotionLangTransformer) throws {
    if event == .epochEnd {
        Context.local.learningPhase = .inference
        var _model = model
        _model.move(to: Device.defaultTFEager)
        for sample in samplesToDecode {
            greedyDecodeSample(sample["sampleID"] as! Int, maxLength: 20, model: _model, device: Device.defaultTFEager)
        }
        _model.move(to: device)
    }
}

// Training loop
optimizerOpts.stepsPerEpoch = dataset.motionSamples.count/batchSize
print("stepsPerEpoch: \(optimizerOpts.stepsPerEpoch)")

print("\nTraining Transformer for the Motion2lang task!")
let statsRecorder = StatsRecorder(logdirURL: rundirURL)
let trainingProgress = TrainingProgress(metrics: [.loss])
let optimizerWrapper = OptimizerWrapper(opts: optimizerOpts, model: model)
var trainingLoop: TrainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizerWrapper.optimizer,
    lossFunction:  embeddedSoftmaxCrossEntropy,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater, decodeSamplesAfterEpoch, saveCheckpoint]
)

try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")
