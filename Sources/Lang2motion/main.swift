import TensorFlow
import TextModels
import TranslationModels
import Foundation
import FoundationXML
import ModelSupport
import Datasets
import SummaryWriter
import LangMotionModels
import TrainingLoop

/// Set training params
let runName = "run_7"
// let batchSize = 4
let batchSize = 150
let maxTextSequenceLength =  20
let maxMotionLength =  100
let nEpochs = 20
let learningRate: Float = 5e-4
let datasetSize: DatasetSize = .full

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")
print("datasetSize: \(datasetSize)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/\(runName)", isDirectory: true)
let checkpointURL = logdirURL.appendingPathComponent("checkpoints", isDirectory: true)
try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)

/// Select eager or X10 backend
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print(device)

// TODO: make sure X10 training works on Colab
// /// X10 warm-up
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
let vocabSize = vocabulary.count
let nbJoints = 47 // TODO: get value from dataset
let nbMixtures = 20
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1

var model = LangMotionTransformer(
    vocabSize: vocabSize, 
    nbJoints: nbJoints,
    nbMixtures: nbMixtures,
    layerCount: layerCount,
    modelSize: modelSize,
    feedForwardSize: feedForwardSize,
    headCount: headCount,
    dropoutProbability: dropoutProbability
)

// TODO: make sure resuming training works again
/// load model checkpoint
// let config = LangMotionTransformerConfig(
//     vocabSize: vocabSize,
//     nbJoints: nbJoints,
//     nbMixtures: nbMixtures,
//     layerCount: layerCount,
//     modelSize: modelSize,
//     feedForwardSize: feedForwardSize,
//     headCount: headCount,
//     dropoutProbability: dropoutProbability
// )

// print("checkpointURL: \(checkpointURL.path)")
// var model = try! LangMotionTransformer(checkpoint: checkpointURL, config: config, name: "model.e17")

/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    trainTestSplit: 1.0,
    device: device
) { (motionSample: MotionSample) -> LangMotionBatch in    
    let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
    let (target2, motionPart) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength)
    let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
    let singleBatch = LangMotionBatch(data: source,label: target2)
    return singleBatch
}

print("Dataset acquired.")

// TODO: make possible to call greedyDecodeMotion() during training again
public func greedyDecodeMotion(sentence: String, prefix: String = "prefix", saveMotion: Bool = true) {
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let processedSentence = textProcessor.preprocess(sentence: sentence, maxTextSequenceLength: maxTextSequenceLength)
    processedSentence.printSentence()

    let decodedMotion = MotionDecoder.greedyDecodeMotion(sentence: processedSentence, transformer: model, nbJoints: nbJoints, nbMixtures: nbMixtures, maxMotionLength: maxMotionLength)
    print("  decodedMotion: min: \(decodedMotion.min()), max: \(decodedMotion.max())")
    let descaledMotion = dataset.scaler.inverse_transform(decodedMotion)
    print("  descaledMotion.shape: \(descaledMotion.shape)")
    print("  descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")

    var imageURL: URL? = dataURL.appendingPathComponent("motion_images/\(prefix).png")
    if !saveMotion { imageURL = nil }
    // TODO: use joint groupping
    motionToImg(url: imageURL, motion: descaledMotion, motionFlag: nil, padTo: maxMotionLength, descr: "\(prefix), \(sentence)", cmapRange: 2.0)

    if saveMotion {
        print("Saved image: \(imageURL!.path)")
        let jointNames = dataset.motionSamples[0].jointNames
        let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
        let mmmURL = dataURL.appendingPathComponent("motion_images/\(prefix).mmm.xml")
        try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
        print("Saved motion: \(mmmURL.path)")
    }
}

// greedyDecodeMotion(sentence: "human is walking", prefix: "foo10", saveMotion: true)
// exit(0)

/// Optimizer
var optimizer = Adam(for: model, learningRate: learningRate)

// Loss function
let args = LossArgs(
        nb_joints: nbJoints,
        nb_mixtures: nbMixtures,
        mixture_regularizer_type: "None",  // ["cv", "l2", "None"]
        mixture_regularizer: 0.0,
        device: device
)

@differentiable
func embeddedNormalMixtureSurrogateLoss(y_pred: MixtureModelPreds, y_target: LangMotionBatch.Target) -> Tensor<Float> {
    // TODO: create tensor on device
    let y_true = TargetTruth(copying: TargetTruth(motion: y_target.targetTruth, stops: y_target.targetTruthStop), to: device)
    let loss = normalMixtureSurrogateLoss(y_true: y_true, y_pred: y_pred, args: args)
    let n_items: Float = Float(loss.shape[0] * loss.shape[1])
    let avg_loss = loss.sum() / n_items
    return avg_loss
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

class StatsRecorder {
    let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)
    var trainingStepCount = 0
    var trainingBatchCount = 0
    var trainingLossSum: Float = 0.0

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
            guard let epochIndex = loop.epochIndex else {
                return
            }
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
  callbacks: [trainingProgress.update, statsRecorder.writeStats])

print("\nTraining Transformer for the Lang2motion task!")
// FIXME: epoch loop workaround for checkpoint saving
for epochIndex in 0..<nEpochs {
    print("epoch \(epochIndex+1)/\(nEpochs)")
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
