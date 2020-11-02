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
let runName = "run_125"
let batchSize = 93
let maxTextSequenceLength =  40
let maxMotionLength =  50
let nEpochs = 100
let multiplyFactor = 100
let discreteBins = 300
let lrSlopeMultiplier: Float = 1.1
let fixedPeekLR: Bool = true
let peakLearningRate: Float = 1e-3
let useBiasCorrection: Bool = true
let weightDecayRate: Float = 0.0001
let beta2: Float = 0.99
let dropoutProbability: Double = 0.0

// peek LR for new training: 1e-3, for resuming: 5e-4
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

let datasetSize: DatasetSize = .small_midi

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(optimizerOpts.peakLearningRate)")
print("datasetSize: \(datasetSize)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/", isDirectory: true)
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
    maxMotionLength: 50,
    multiplyFactor: multiplyFactor,
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
    nbMixtures: 20,
    layerCount: 6,
    encoderDepth: 256,
    decoderDepth: 512,
    feedForwardSize: 2048,
    headCount: 16,
    dropoutProbability: dropoutProbability,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500,
    discreteBins: discreteBins,
    activation: swish
)

/// create new model
 var model = LangMotionCatDistTransformer(config: config)

/// load model checkpoint
//var model = try! LangMotionCatDistTransformer(checkpoint: logdirURL.appendingPathComponent("run_113/checkpoints"), config: config, name: "model.e100")

// Loss function
let args = CDLossArgs(
        device: device
)

@differentiable(wrt: y_pred)
func embeddedCategoryDistributionSurrogateLoss(y_pred: LangMotionCatDistTransformerOutput<Float>, y_true: LangMotionBatch.Target) -> Tensor<Float> {
    return categoryDistributionSurrogateLoss(y_pred: y_pred.preds, y_true: y_true, args: args)
}

/// Set up decoding
// TODO: make possible to call greedyDecodeMotion() during training again
public func greedyDecodeMotion(dataset: Lang2Motion, model: LangMotionTransformer,
                               sentence: String, prefix: String = "prefix", saveMotion: Bool = true, motionsURL: URL?) {
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let processedSentence = textProcessor.preprocess(sentence: sentence, maxTextSequenceLength: maxTextSequenceLength)
    processedSentence.printSentence()

    let (decodedMotion, decodedMotionFlag) = MotionDecoder.greedyDecodeMotion(sentence: processedSentence, startMotion: nil, transformer: model, maxMotionLength: maxMotionLength)
    print("  decodedMotion: min: \(decodedMotion.min()), max: \(decodedMotion.max())")
    let descaledMotion = dataset.scaler.inverse_transform(decodedMotion)
    print("  descaledMotion.shape: \(descaledMotion.shape)")
    print("  descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")
    var imageURL: URL? = nil
    
    if !saveMotion { imageURL = nil } else {
        imageURL = motionsURL!.appendingPathComponent("\(prefix).png")
    }
    // use joint groupping
    let grouppedJointsMotion = MotionSample.grouppedJoints(motion: descaledMotion, jointNames: dataset.motionSamples[0].jointNames)
    let _ = motionToImg(url: imageURL, motion: grouppedJointsMotion, motionFlag: decodedMotionFlag, padTo: maxMotionLength, descr: "\(sentence)", cmapRange: 2.0)

    if saveMotion {
        print("Saved image: \(imageURL!.path)")
        let jointNames = dataset.motionSamples[0].jointNames
        let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
        let mmmURL = motionsURL!.appendingPathComponent("\(prefix).mmm.xml")
        try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
        print("Saved motion: \(mmmURL.path)")
    }
}

public func saveCheckpoint<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: LangMotionTransformer) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
    }
}

// Training loop
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
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater]
)

try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

//try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")
