import TensorFlow
import TextModels
import TranslationModels
import Foundation
import FoundationXML
import ModelSupport
import Datasets
import SummaryWriter
import LangMotionModels

/// Set training params
let runName = "run_4"
// let batchSize = 4
let batchSize = 150
let maxTextSequenceLength =  20
let maxMotionLength =  100
let nEpochs = 40
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")


let datasetSize: DatasetSize = .full

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Lang2motion/\(runName)", isDirectory: true)
let checkpointURL = logdirURL.appendingPathComponent("checkpoints", isDirectory: true)
try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)

/// Select eager or X10 backend
// let device = Device.defaultXLA
let device = Device.defaultTFEager
print(device)

// /// X10 warm-up
// let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
// let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
// let eagerTensorSum = eagerTensor1 + eagerTensor2
// print(eagerTensorSum)
// print(eagerTensor1.device)
// let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
// print(x10Tensor2.device)

/// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor2(vocabulary: vocabulary, tokenizer: tokenizer, maxTextSequenceLength: maxTextSequenceLength, maxMotionLength: maxMotionLength)

/// instantiate model
let vocabSize = vocabulary.count
let nbJoints = 47 // TODO: get value from dataset
let nbMixtures = 20
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1

// var model = LangMotionTransformer(
//     vocabSize: vocabSize, 
//     nbJoints: nbJoints,
//     nbMixtures: nbMixtures,
//     layerCount: layerCount, 
//     modelSize: modelSize, 
//     feedForwardSize: feedForwardSize, 
//     headCount: headCount, 
//     dropoutProbability: dropoutProbability
// )

/// load model checkpoint
let config = LangMotionTransformerConfig(
    vocabSize: vocabSize,
    nbJoints: nbJoints,
    nbMixtures: nbMixtures,
    layerCount: layerCount,
    modelSize: modelSize,
    feedForwardSize: feedForwardSize,
    headCount: headCount,
    dropoutProbability: dropoutProbability
)

print("checkpointURL: \(checkpointURL.path)")
var model = try! LangMotionTransformer(checkpoint: checkpointURL, config: config, name: "model.e17")
model.move(to: device)

/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    trainTestSplit: 1.0
) { (example: MotionSample2) -> LangMotionBatch in    
    let singleBatch = textProcessor.preprocess(example: example)
    return singleBatch
}

print("Dataset acquired.")

/// one example to single batch
// print("\nSingle batch")
// print("============")
// let example = dataset.trainExamples[0]
// print("example.sentence: \"\(example.sentence)\"")

// let singleBatch = textProcessor.preprocess(example: example)
// LangMotionBatch.printBatch(singleBatch)

/// Test model with one batch
/// get a batch
// print("\nOne batch:")
// print("=========")
// var epochIterator = dataset.trainEpochs.enumerated().makeIterator()
// let epoch = epochIterator.next()
// let batches = Array(epoch!.1)
// let batch: LangMotionBatch = batches[0]
// printBatch(batch)

/// run one batch
// print("\nRun one batch:")
// print("==============")
// let deviceBatch = LangMotionBatch(copying: batch, to: device)
// let batch_generated = model.generate(input: deviceBatch)
// print("batch_generated.shape: \(batch_generated.shape)")

/// decode single batch
// print("\nDecode single batch:")
// print("====================")
// let singlePreds: MixtureModelPreds = model.generate(input: LangMotionBatch(copying: singleBatch, to: device))
// singlePreds.printPreds()

// let (motion, log_probs, done) = performNormalMixtureSampling(
//     preds: single_generated, nb_joints: nbJoints, nb_mixtures: nbMixtures, maxMotionLength: maxMotionLength)

// let descaled_motion = dataset.scaler.inverse_transform(motion)

// print("motion.shape: \(motion.shape)")
// print("log_probs.count: \(log_probs.count)")
// print("done.shape: \(done.shape)")
// print("done: \(done)")
// print("log_probs: \(log_probs)")
// print("descaled_motion: \(descaled_motion)")

// motionToImg(url: dataURL.appendingPathComponent("motion_images/foo8.png"), 
//             motion: motion, motionFlag: done, padTo: maxMotionLength, descr: "\(example.sentence)")

// motionToImg(url: dataURL.appendingPathComponent("motion_images/foo8_descaled.png"), 
//             motion: descaled_motion, motionFlag: done, padTo: maxMotionLength, descr: "\(example.sentence)")

public func greedyDecodeMotion(sentence: String, prefix: String = "prefix", saveMotion: Bool = true) {
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let source = textProcessor.preprocess(sentence: sentence)
    source.printSource()

    let decodedMotion = MotionDecoder.greedyDecodeMotion(source: source, transformer: model, nbJoints: nbJoints, nbMixtures: nbMixtures, maxMotionLength: maxMotionLength)
    print("  decodedMotion: min: \(decodedMotion.min()), max: \(decodedMotion.max())")
    let descaledMotion = dataset.scaler.inverse_transform(decodedMotion)
    print("  descaledMotion.shape: \(descaledMotion.shape)")
    print("  descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")

    var imageURL: URL? = dataURL.appendingPathComponent("motion_images/\(prefix).png")
    if !saveMotion { imageURL = nil }
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
optimizer = Adam(copying: optimizer, to: device)

let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

let args = LossArgs(
        nb_joints: nbJoints,
        nb_mixtures: nbMixtures,
        mixture_regularizer_type: "None",  // ["cv", "l2", "None"]
        mixture_regularizer: 0.0
)

/// Training helpers
func update(model: inout LangMotionTransformer, using optimizer: inout Adam<LangMotionTransformer>, for batch: LangMotionBatch) -> Float {
    let y_true = TargetTruth(motion: batch.targetTruth, stops: batch.targetTruthStop)
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            (model) -> Tensor<Float> in
            let y_pred = model.generate(input: batch)
            let loss = normalMixtureSurrogateLoss(y_true: y_true, y_pred: y_pred, args: args)
            let n_items: Float = Float(loss.shape[0] * loss.shape[1])
            // let ones = Tensor<Float>(ones: loss.shape)
            // let nans = loss.isNaN
            // let loss_notNaN = loss.replacing(with:ones, where:nans)
            // let avg_loss = loss_notNaN.sum() / n_items
            let avg_loss = loss.sum() / n_items
            // print("avg_loss: \(avg_loss)")
            return avg_loss
        }
        optimizer.update(&model, along: grad)
        LazyTensorBarrier()
        return loss.scalarized()
    }
    return result
}

func validate(model: inout LangMotionTransformer, for batch: LangMotionBatch) -> Float {
    let y_true = TargetTruth(motion: batch.targetTruth, stops: batch.targetTruthStop)
    let result = withLearningPhase(.inference) { () -> Float in
        let y_pred = model.generate(input: batch)
        let loss = normalMixtureSurrogateLoss(y_true: y_true, y_pred: y_pred, args: args)
        let n_items: Float = Float(loss.shape[0] * loss.shape[1])
        let avg_loss = loss.sum() / n_items
        return avg_loss.scalarized()
    }
    LazyTensorBarrier()
    return result
}

/// Training loop
print("\nTraining Transformer for the Lang2motion task!")
var trainingStepCount = 0
let print_every = 50
let limit_print_to_step = 5
let start_epoch = 17
var current_epoch = 0
time() {
    LazyTensorBarrier()
    for (epoch, epochBatches) in dataset.trainEpochs.prefix(nEpochs).enumerated() {
        current_epoch = start_epoch + epoch + 1
        print("[Epoch \(current_epoch)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        for eagerBatch in epochBatches {
            // if (trainingStepCount < limit_print_to_step || trainingStepCount % print_every == 0) {
            //     print("==> step \(trainingStepCount)")
            // }
            let batch = LangMotionBatch(copying: eagerBatch, to: device)
            let loss: Float = update(model: &model, using: &optimizer, for: batch)
            if (trainingStepCount < limit_print_to_step || trainingStepCount % print_every == 0) {
                print("current loss at step \(trainingStepCount): \(loss)")
            }
            trainingLossSum += loss
            trainingBatchCount += 1
            summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: trainingLossSum / Float(trainingBatchCount))
            trainingStepCount += 1
        }
        print(
            """
            Training loss: \(trainingLossSum / Float(trainingBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: current_epoch, value: trainingLossSum / Float(trainingBatchCount))

        if epoch == 0 {
            print("dataset.testBatches.count: \(dataset.testBatches.count)")
        }
        Context.local.learningPhase = .inference
        var devLossSum: Float = 0
        var devBatchCount = 0
        var totalGuessCount = 0

        for eagerBatch in dataset.testBatches {
            let batch = LangMotionBatch(copying: eagerBatch, to: device)
            let loss: Float = validate(model: &model, for: batch)
            let valBatchSize = batch.target.motion.shape[0]

            devLossSum += loss
            devBatchCount += 1
            totalGuessCount += valBatchSize
        }

        print(
            """
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: current_epoch, value: devLossSum / Float(devBatchCount))
        try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(current_epoch)")
        if current_epoch >= 2 {
            // greedyDecodeMotion(sentence: "human walks and then runs and later sits down", prefix: "epoch_\(current_epoch)", saveMotion: true)
        }
    }
    summaryWriter.flush()
}

print("\nFinished training.")
