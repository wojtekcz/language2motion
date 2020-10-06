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
let runName = "run_63"
let batchSize = 20
let maxTextSequenceLength =  40
let maxMotionLength =  150
let nEpochs = 4

var optimizerOpts = OptimizerOpts(
    peakLearningRate: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    useBiasCorrection: false,
    lrSlopeMultiplier: 2,
    nEpochs: nEpochs
)

let datasetSize: DatasetSize = .multi_mini

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
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")

/// instantiate text processor
print("instantiate text processor")
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)


/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 20,
    maxMotionLength: 150,
    trainTestSplit: 0.9,
    device: device
) { (motionSample: MotionSample) -> LangMotionBatch in
    let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
    let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength)

    let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
    let singleBatch = LangMotionBatch(data: source, label: target)
    return singleBatch
}

print("Dataset acquired.")

/// instantiate model
print("instantiate model")
let config = LangMotionTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47,
    nbMixtures: 20,
    layerCount: 6,
    encoderDepth: 64,
    decoderDepth: 128,
    feedForwardSize: 512,
    headCount: 4,
    dropoutProbability:  0.1,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500,
    encoderSelfAttentionTemp: 1,
    decoderSourceAttentionTemp: 1,
    decoderSelfAttentionTemp: 1
)

/// create new model
var model = LangMotionTransformer(config: config)

/// load model checkpoint
//var model = try! LangMotionTransformer(checkpoint: logdirURL.appendingPathComponent("run_61/checkpoints"), config: config, name: "model.e2")

// Loss function
let args = LossArgs(
        nb_joints: config.nbJoints,
        nb_mixtures: config.nbMixtures,
        mixture_regularizer_type: "None",  // ["cv", "l2", "None"]
        mixture_regularizer: 0.0,
        device: device
)

@differentiable(wrt: y_pred)
func embeddedNormalMixtureSurrogateLoss(y_pred: LangMotionTransformerOutput<Float>, y_true: LangMotionBatch.Target) -> Tensor<Float> {
    return normalMixtureSurrogateLoss(y_pred: y_pred.preds, y_true: y_true, args: args)
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

    let decodedMotion = MotionDecoder.greedyDecodeMotion(sentence: processedSentence, transformer: model, nbJoints: config.nbJoints, nbMixtures: config.nbMixtures, maxMotionLength: maxMotionLength)
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
    motionToImg(url: imageURL, motion: grouppedJointsMotion, motionFlag: nil, padTo: maxMotionLength, descr: "\(sentence)", cmapRange: 2.0)

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
    lossFunction: embeddedNormalMixtureSurrogateLoss,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, optimizerWrapper.learningRateUpdater, saveCheckpoint]
)

try! trainingLoop.fit(&model, epochs: nEpochs, on: device)

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")
