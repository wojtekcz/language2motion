import TensorFlow
import TextModels
import TranslationModels
import Foundation
#if os(Linux)
  import FoundationXML
#endif
import ModelSupport
import Datasets
import SummaryWriter
import LangMotionModels
import TrainingLoop
import x10_optimizers_optimizer

/// Set training params
let runSetName = "run_set_47"
let batchSize = 100
let maxTextSequenceLength =  40
let maxMotionLength =  50
let nEpochs = 100

let datasetSize: DatasetSize = .small_multi_midi
let multiplyFactor = 10
let lrSlopeMultiplier: Float = 1.1
let fixedPeekLR: Bool = true

let commonRunsSettings: [String:Any] = [
    "dropout": 0.0, "beta1": 0.9, "beta2": 0.99, "wd": 0.01, "useBiasCorrection": true,
]

// peek LR for new training: 1e-3, for resuming: 5e-4 (for full dataset)
let runsSettings: [[String:Any]] = [
    ["lr": 1e-4],
]

//print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
//print("peakLearningRate: \(optimizerOpts.peakLearningRate)")
print("datasetSize: \(datasetSize)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
let _ = _ExecutionContext.global

/// Select eager or X10 backend
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")

/// instantiate text processor
print("instantiate text processor")
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)

let logdirURL = dataURL.appendingPathComponent("runs/Lang2motionSet/", isDirectory: true)
let runSetURL = logdirURL.appendingPathComponent(runSetName, isDirectory: true)
let checkpointURL = runSetURL.appendingPathComponent("checkpoints", isDirectory: true)

#if os(Linux)
    try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
#endif
/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 10,
    maxMotionLength: 50,
    multiplyFactor: multiplyFactor,
    trainTestSplit: 1.0,
    device: device
) { (motionSample: MotionSample) -> LangMotionBatch in
    let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
    let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength)

    let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
    let singleBatch = LangMotionBatch(data: source, label: target)
    return singleBatch
}

print("Dataset acquired.")

// vars used in extra.swift
let nbJoints = 47
let nbMixtures = 20
var runName = ""

print("\nTraining Transformer for the Lang2motion task!")
for runNum in 0..<runsSettings.count {
    var runSettings = runsSettings[runNum]
    runSettings = runSettings.merging(commonRunsSettings, uniquingKeysWith: { (first, _) in first })
    print("runNum: \(runNum+1), runSettings: \(runSettings)")
    let peakLearningRate = Float(runSettings["lr"] as! Double)
    let dropoutProbability = runSettings["dropout"] as! Double
    let weightDecayRate = Float(runSettings["wd"] as! Double)
    let beta1 = Float(runSettings["beta1"] as! Double)
    let beta2 = Float(runSettings["beta2"] as! Double)
    let useBiasCorrection = runSettings["useBiasCorrection"] as! Bool
    
    // runName = "run_\(runNum+1)_wd_\(weightDecayRate)"
    // runName = "run_\(runNum+1)_bcor_\(useBiasCorrection)"
    // runName = "run_\(runNum+1)_beta2_\(beta2)"
    runName = "run_\(runNum+1)_lr_\(peakLearningRate)"
    let rundirURL = runSetURL.appendingPathComponent(runName, isDirectory: true)
    
    let config = LangMotionTransformerConfig(
        vocabSize: vocabulary.count, nbJoints: 47, nbMixtures: 20,
        layerCount: 6,
        encoderDepth: 256, decoderDepth: 512, feedForwardSize: 2048,
        headCount: 16,
        dropoutProbability: dropoutProbability,
        sentenceMaxPositionalLength: 100, motionMaxPositionalLength: 500, mixtureDepth: 1500
    )

     var model = LangMotionTransformer(config: config)
//    var model = try! LangMotionTransformer(checkpoint: logdirURL.appendingPathComponent("run_set_38/checkpoints"), config: config, name: "run_1_lr_0.0001.e98")

    var optimizerOpts = OptimizerOpts(
        peakLearningRate: peakLearningRate,
        beta1: beta1, beta2: beta2,
        weightDecayRate: weightDecayRate, // default 0.01
        useBiasCorrection: useBiasCorrection, lrSlopeMultiplier: lrSlopeMultiplier, nEpochs: nEpochs,
        fixedPeekLR: fixedPeekLR
    )
    optimizerOpts.stepsPerEpoch = dataset.motionSamples.count/batchSize

    let statsRecorder = StatsRecorder(logdirURL: rundirURL)
    let trainingProgress = TrainingProgress(metrics: [.loss])
    let optimizerWrapper = OptimizerWrapper(opts: optimizerOpts, model: model)
    var trainingLoop = TrainingLoop(
        training: dataset.trainEpochs,
        validation: dataset.testBatches,
        optimizer: optimizerWrapper.optimizer,
        lossFunction: embeddedNormalMixtureSurrogateLoss,
        callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater, saveCheckpoint]
    )

    try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

    try! model.writeCheckpoint(to: checkpointURL, name: "\(runName).final")
}

print("\nFinished trainings.")
