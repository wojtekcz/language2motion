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
let runName = "run_52"
// let batchSize = 10
let batchSize = 50
let maxTextSequenceLength =  20
let maxMotionLength =  50
let nEpochs = 30
let peakLearningRate: Float = 1e-3 // bs=50
// let peakLearningRate: Float = 2e-4 // bs=10
// let peakLearningRate: Float = 2e-5

let stepsPerEpoch = 1967/batchSize*2 // function of training set size and batching configuration

let beta1: Float = 0.9
let beta2: Float = 0.999
let useBiasCorrection = false

let datasetSize: DatasetSize = .full

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(peakLearningRate)")
print("datasetSize: \(datasetSize)")
print("stepsPerEpoch: \(stepsPerEpoch)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/\(runName)", isDirectory: true)
let checkpointURL = logdirURL.appendingPathComponent("checkpoints", isDirectory: true)

// FIXME: how to make macOS builds use filesystem in read/write mode?
#if os(Linux)
    try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
#endif

/// Select eager or X10 backend
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")

// TODO: make sure X10 training works on Colab
/// X10 warm-up
// let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
// let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
// let eagerTensorSum = eagerTensor1 + eagerTensor2
// print(eagerTensorSum)
// print(eagerTensor1.device)
// let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
// print(x10Tensor2.device)

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
// let _ = _ExecutionContext.global

/// instantiate text processor
print("instantiate text processor")
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)

/// instantiate model
print("instantiate model")
let modelSize = 128
let config = LangMotionTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47, // TODO: get value from dataset
    nbMixtures: 20,
    layerCount: 6,
    modelSize: modelSize,
    feedForwardSize: 512,
    headCount: 4,
    dropoutProbability:  0.1,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500,
    motionPositionalEncodingSize: 32,
    encoderSelfAttentionTemp: sqrt(Double(modelSize)),
    decoderSourceAttentionTemp: sqrt(Double(modelSize)),
    decoderSelfAttentionTemp: Double(modelSize)
)

var start_epoch = 0

/// create new model
var model = LangMotionTransformer(config: config)

/// load model checkpoint
// print("checkpointURL: \(checkpointURL.path)")
// start_epoch = 2
// var model = try! LangMotionTransformer(checkpoint: checkpointURL, config: config, name: "model.e\(start_epoch)")

/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 20,
    maxMotionLength: 50,
    trainTestSplit: 1.0,
    device: device
) { (motionSample: MotionSample) -> LangMotionBatch in    
    let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
    let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, shiftMaskRight: true)

    let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
    let singleBatch = LangMotionBatch(data: source, label: target)
    return singleBatch
}

print("Dataset acquired.")

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

// greedyDecodeMotion(sentence: "human is walking", prefix: "foo10", saveMotion: true)
// exit(0)

/// Optimizer
// var optimizer = Adam(for: model, learningRate: learningRate)

var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
    for: model,
    TensorVisitorPlan(model.differentiableVectorView),
    defaultOptimizer: makeWeightDecayedAdam(
      learningRate: peakLearningRate,
      beta1: beta1,
      beta2: beta2
    )
)

var scheduledLearningRate = LinearlyDecayedParameter(
  baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter<Float>(peakLearningRate),
      warmUpStepCount: 20,
      warmUpOffset: 0),
  slope: -(peakLearningRate / Float(stepsPerEpoch * nEpochs)),  // The LR decays linearly to zero.
  startStep: 10
)

public func learningRateUpdater<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    if event == .updateStart {
        let optimizer: GeneralOptimizer<LangMotionTransformer> = loop.optimizer as! GeneralOptimizer<LangMotionTransformer>
        let step = optimizer.step + 1 // for scheduled rates and bias correction, steps start at 1
        optimizer.learningRate = scheduledLearningRate(forStep: UInt64(step))
        if useBiasCorrection {
          let f_step = Float(step)
          optimizer.learningRate *= sqrtf(1 - powf(beta2, f_step)) / (1 - powf(beta1, f_step))
        }
        // print("\noptimizer: step: \(optimizer.step), learningRate: \(optimizer.learningRate)")
    }
}

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

public func saveCheckpoint<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    if event == .epochEnd {
        guard let epochIndex = loop.epochIndex else {
            return
        }
        let transformer: LangMotionTransformer = loop.model as! LangMotionTransformer
        try! transformer.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
    }
}

public class StatsRecorder {
    let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)
    public var trainingStepCount = 0
    public var trainingBatchCount = 0
    public var trainingLossSum: Float = 0.0
    public var epochIndex = 0 // FIXME: Workaround

    public func writeStats<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
        if event == .batchEnd {
            guard 
            // let batchIndex = loop.batchIndex, 
            let trainingLoss = loop.lastLoss else {
                return
            }
            // print("\nbatch stats: batchIndex: \(batchIndex), trainingStepCount: \(trainingStepCount), trainingLoss: \(trainingLoss)")
            summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value:trainingLoss.scalar!)
            trainingStepCount += 1
            trainingBatchCount += 1
            trainingLossSum += Float(trainingLoss.scalar!)
        }
        if event == .epochStart {
            trainingBatchCount = 0
            trainingLossSum = 0.0
        }
        if event == .epochEnd {
            // guard let epochIndex = loop.epochIndex else {
            //     return
            // }
            let current_epoch = epochIndex + 1
            let epochTrainingLoss = trainingLossSum / Float(trainingBatchCount)
            // print("\nepoch stats: current_epoch: \(current_epoch), epochTrainingLoss: \(epochTrainingLoss)")
            summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: current_epoch, value: epochTrainingLoss)
        }
        if event == .fitEnd {
            summaryWriter.flush()
        }
    }
}

let statsRecorder = StatsRecorder()

// Training loop
print("\nSetting up the training loop")
let trainingProgress = TrainingProgress(metrics: [.loss])
var trainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizer,
    lossFunction: embeddedNormalMixtureSurrogateLoss,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, learningRateUpdater]
    // callbacks: []
)

print("\nTraining Transformer for the Lang2motion task!")
// FIXME: epoch loop workaround for checkpoint saving
for epochIndex in start_epoch..<start_epoch+nEpochs {
    print("epoch \(epochIndex+1)/\(start_epoch + nEpochs)")
    statsRecorder.epochIndex = epochIndex
    try! trainingLoop.fit(&model, epochs: 1, on: device)
    try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")
}

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")

// TODO: time 1 epoch training

// time() {
//     for (epoch, epochBatches) in dataset.trainEpochs.prefix(nEpochs).enumerated() {
//         if current_epoch >= 2 {
//             // greedyDecodeMotion(sentence: "human walks and then runs and later sits down", prefix: "epoch_\(current_epoch)", saveMotion: true)
//         }
//     }
// }
