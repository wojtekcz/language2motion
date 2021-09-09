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
import MotionGenerator

/// Set training params
let maxSamples: Int? = nil

let maxSamplesStr = maxSamples != nil ? "_\(maxSamples!)" : ""

let runName = "run_185" //_maxSamples\(maxSamplesStr)"
let batchSize = 100
let maxTextSequenceLength =  40
let maxMotionLength = 75
let nEpochs = 150
let multiplyFactor = 4 //17
let discreteBins = 300
let lrSlopeMultiplier: Float = 1.05
let fixedPeekLR: Bool = false
let peakLearningRate: Float = 5e-2 //5e-3
let useBiasCorrection: Bool = true
let weightDecayRate: Float = 0.001
let beta2: Float = 0.99
let dropoutProbability: Double = 0.0

let datasetSize: DatasetSize = .full

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(peakLearningRate)")
print("datasetSize: \(datasetSize)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdir = "runs/Lang2motion/"
let logdirURL = dataURL.appendingPathComponent(logdir, isDirectory: true)
let rundirURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
let checkpointURL = rundirURL.appendingPathComponent("checkpoints", isDirectory: true)

#if os(Linux)
    try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
#endif

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
let _ = _ExecutionContext.global

/// Select eager or X10 backend
//let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")

/// instantiate text processor
print("instantiate text processor")
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
var discretizer = MotionDiscretizer(n_bins: discreteBins)


/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 10,
    maxMotionLength: maxMotionLength,
    multiplyFactor: multiplyFactor,
    maxSamples: maxSamples,
    discretizer: &discretizer,
    trainTestSplit: 1.0,
    device: device
) { (motionSample: MotionSample) -> LangMotionBatch in
    let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
    let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer)

    let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
    let singleBatch = LangMotionBatch(data: source, label: target)
    return singleBatch
}

print("Dataset acquired.")

/// instantiate model
print("instantiate model")
let config = LangMotionCatDistTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47,
    layerCount: 12,
    encoderDepth: 64,
    decoderDepth: 320,
    feedForwardSize: 2560,
    headCount: 16,
    dropoutProbability: dropoutProbability,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500,
    discreteBins: discreteBins,
    activation: .swish
)

/// create new model
var model = LangMotionCatDistTransformer(config: config)

/// load model checkpoint
// var model = try! LangMotionCatDistTransformer(checkpoint: logdirURL.appendingPathComponent("run_147/checkpoints"), config: config, name: "model.e23")

// Loss function
let args = CDLossArgs(
        device: device
)

@differentiable(wrt: y_pred)
func embeddedCategoryDistributionSurrogateLoss(y_pred: LangMotionCatDistTransformerOutput<Float>, y_true: LangMotionBatch.Target) -> Tensor<Float> {
    return categoryDistributionSurrogateLoss(y_pred: y_pred.preds, y_true: y_true, args: args)
}

public func saveCheckpoint<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: LangMotionCatDistTransformer) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
    }
}

//let samplesToDecode: [[String:Any]] = [
//   ["sampleID": 449, "text": "A person runs forward."],
//   ["sampleID": 3921, "text": "A human is swimming."],
//   ["sampleID": 843, "text": "A person walks."],
//   ["sampleID": 1426, "text": "A person plays the air guitar."],
//   ["sampleID": 1292, "text": "A person performs a squat."],
//   ["sampleID": 1315, "text": "A human raises their left foot and touches it with the right hand."]
//]

// micro dataset
let samplesToDecode: [[String:Any]] = [
    ["sampleID": 1886, "text": "A human is being pushed."],
    ["sampleID": 106, "text": "A person walks forwards."],
    ["sampleID": 1364, "text": "A person hits something."],
    ["sampleID": 1379, "text": "A person waves with the right hand."],
    ["sampleID": 1425, "text": "A person plays the guitar."],
    ["sampleID": 2001, "text": "A person turns 180 degrees on his left foot."],
    ["sampleID": 576, "text": "A person walks and turns left"],
    ["sampleID": 1034, "text": "A person is walking forward"],
    ["sampleID": 2083, "text": "A person goes down to his knees."],
    ["sampleID": 738, "text": "A person walking a left curve"],
    ["sampleID": 1291, "text": "Somone is raising his fists in front of his upper body and doing a low kick with his left foot."],
    ["sampleID": 2114, "text": "Someone is kneeling down."],
    ["sampleID": 67, "text": "A human walks in a counter-clockwise circle."]
]

let mgMgr = MotionGenerationManager(scaler: dataset.scaler, jointNames: dataset.motionSamples[0].jointNames, textProcessor: textProcessor, discretizer: discretizer, logdir: logdir, runName: runName)

func generateMotion(sentence: String, prefix: String, model: LangMotionCatDistTransformer) {
    let bestLogProbs = true
    let fixRotation = true
    let saveMMM = true
    let maxMotionLength = 50

    let opts = GenOpts(nSamples: 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMMM, encoderSelfAttentionTemp: 1.0, decoderSourceAttentionTemp: 1.0, decoderSelfAttentionTemp: 1.0, maxMotionLength: maxMotionLength, sentence: sentence)
    
    let _ = mgMgr.generateMotion(genOpts: opts, prefix: prefix, model: model)
}

public func generateMotions<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: LangMotionCatDistTransformer) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        let prefix = "model.e\(epochIndex+1)"
        print("generateMotions() for \"\(prefix)\"")

        Context.local.learningPhase = .inference
        // var _model = model
        // _model.move(to: Device.defaultTFEager)
        for sample in samplesToDecode {
            generateMotion(sentence: sample["text"] as! String, prefix: "\(prefix).\(sample["sampleID"]!)", model: model)
        }
        // _model.move(to: device)
    }
}

// Training loop

var optimizerOpts = OptimizerOpts(
    peakLearningRate: peakLearningRate,
    beta1: 0.9,
    beta2: beta2,
    weightDecayRate: weightDecayRate, // default 0.01
    useBiasCorrection: useBiasCorrection,
    lrSlopeMultiplier: lrSlopeMultiplier,
    nEpochs: nEpochs,
    fixedPeekLR: fixedPeekLR
)

optimizerOpts.stepsPerEpoch = dataset.motionSamples.count/batchSize
print("stepsPerEpoch: \(optimizerOpts.stepsPerEpoch)")

print("\nTraining Transformer for the Lang2motion task!")
let statsRecorder = StatsRecorder(logdirURL: rundirURL)
let trainingProgress = TrainingProgress(metrics: [.loss])
let optimizerWrapper = OptimizerWrapper(opts: optimizerOpts, model: model)
var trainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizerWrapper.optimizer,
    lossFunction: embeddedCategoryDistributionSurrogateLoss,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater, generateMotions, saveCheckpoint]
)

try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")
