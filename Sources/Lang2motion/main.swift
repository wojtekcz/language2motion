import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionModels

/// Set training params
let runName = "run_1"
let batchSize = 300
let maxTextSequenceLength =  50
let maxMotionLength =  50
let nEpochs = 10
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

enum DatasetSize: String {
    case full = ""
    case mini = "mini."
    case s490 = "490."
}

let datasetSize: DatasetSize = .s490

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.\(datasetSize.rawValue)plist")
let langDatasetURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

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
let nbJoints = 48 // TODO: get value from dataset
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1

var transformer = LangMotionTransformer(
    vocabSize: vocabSize, 
    nbJoints: nbJoints,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

let nbMixtures = 20
var mixtureModel = MotionGaussianMixtureModel(inputSize: nbJoints, nbJoints: nbJoints, nbMixtures: nbMixtures)
// mixtureModel.move(to: device)

var model = LangMotionModel(transformer: transformer, mixtureModel: mixtureModel)
model.move(to: device)

/// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Motion(
    motionDatasetURL: motionDatasetURL,
    langDatasetURL: langDatasetURL,
    batchSize: batchSize
) { (example: Lang2Motion.Example) -> LangMotionBatch in    
    let singleBatch = textProcessor.preprocess(example: example)
    return singleBatch
}

print("Dataset acquired.")

func printBatch(_ batch: LangMotionBatch) {
    print("type: \(type(of:batch))")
    print("sampleID: shape \(batch.sampleID.shape), value \(batch.sampleID)")

    print("source")
    print("  tokenIds.shape: \(batch.tokenIds.shape)")
    print("  mask.shape: \(batch.mask.shape)")
    print("  tokenCount: shape \(batch.tokenCount.shape), value \(batch.tokenCount)")

    print("target")
    print("  targetMotionFrames.shape: \(batch.targetMotionFrames.shape)")
    print("  targetMask.shape: \(batch.targetMask.shape)")
    print("  targetTruth.shape: \(batch.targetTruth.shape)")
    print("  origMotionFramesCount: shape \(batch.origMotionFramesCount.shape), value \(batch.origMotionFramesCount)")
}

/// one example to single batch
print("\nSingle batch")
print("============")
let example = dataset.trainExamples[0]
print("example.sentence: \"\(example.sentence)\"")

let singleBatch = textProcessor.preprocess(example: example)
printBatch(singleBatch)

/// Test model with one batch
/// get a batch
// print("\nOne batch:")
// print("=========")
// var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
// let epoch = epochIterator.next()
// let batches = Array(epoch!.1)
// let batch: LangMotionBatch = batches[0]
// printBatch(batch)

/// run one batch
// print("\nRun one batch:")
// print("==============")
// let deviceBatch = LangMotionBatch(copying: batch, to: device)
// let decoded = model(deviceBatch)
// print("decoded.shape: \(decoded.shape)")
// let generated = transformer.generate(input: deviceBatch)
// print("generated.shape: \(generated.shape)")
// let allDecoderOutputs = mixtureModel(generated)
// print("allDecoderOutputs.shape: \(allDecoderOutputs.shape)")

// let allDecoderOutputs = model.generate(input: deviceBatch)
// print("allDecoderOutputs.shape: \(allDecoderOutputs.shape)")

/// Optimizer
var optimizer = Adam(for: model, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("tboard/Lang2motion/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

let args = LossArgs(
        nb_joints: nbJoints,
        nb_mixtures: nbMixtures,
        mixture_regularizer_type: "None",  // ["cv", "l2", "None"]
        mixture_regularizer: 0.0
)

/// Training helpers
func update(model: inout LangMotionModel, using optimizer: inout Adam<LangMotionModel>, for batch: LangMotionBatch) -> Float {
    let y_true = batch.targetTruth
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

/// returns validation loss
func validate(model: inout LangMotionModel, for batch: LangMotionBatch) -> Float {
    let y_true = batch.targetTruth
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
let print_every = 10
time() {
    LazyTensorBarrier()
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(nEpochs).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        for eagerBatch in epochBatches {
            if (trainingStepCount < 5 || trainingStepCount % print_every == 0) {
                print("==> step \(trainingStepCount)")
            }
            let batch = LangMotionBatch(copying: eagerBatch, to: device)
            let loss: Float = update(model: &model, using: &optimizer, for: batch)
            if (trainingStepCount < 5 || trainingStepCount % print_every == 0) {
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
        summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: epoch+1, value: trainingLossSum / Float(trainingBatchCount))

        if epoch == 0 {
            print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
        }
        Context.local.learningPhase = .inference
        var devLossSum: Float = 0
        var devBatchCount = 0
        var totalGuessCount = 0

        for eagerBatch in dataset.validationBatches {
            let batch = LangMotionBatch(copying: eagerBatch, to: device)
            let loss: Float = validate(model: &model, for: batch)
            let valBatchSize = batch.targetMotionFrames.shape[0]

            devLossSum += loss
            devBatchCount += 1
            totalGuessCount += valBatchSize
        }

        print(
            """
            totalGuessCount: \(totalGuessCount) \
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch+1, value: devLossSum / Float(devBatchCount))
        // greedyDecodeSample(Int(example.id)!)
    }
    summaryWriter.flush()
}

print("\nFinished training.")
