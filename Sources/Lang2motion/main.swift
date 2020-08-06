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
let runName = "run_1"
let batchSize = 4
// let batchSize = 150
let maxTextSequenceLength =  20
let maxMotionLength =  100
let nEpochs = 10
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("maxMotionLength: \(maxMotionLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")


let datasetSize: DatasetSize = .mini

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")
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
// var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
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

public func greedyDecodeMotion(sentence: String, prefix: String = "prefix", showMotion: Bool = false) {
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let source = textProcessor.preprocess(sentence: sentence)
    source.printSource()

    print("\nEncode:")
    print("======")
    let memory = model.encode(input: source)
    print("  memory.count: \(memory.shape)")

    print("\nGenerate:")
    print("=========")
    // tensor for neutral motion frame
    var ys: Tensor<Float> = Tensor<Float>(repeating:0.0, shape: [1, 1, nbJoints])
    for _ in 0..<maxMotionLength {
        // prepare input
        let targetMask = Tensor<Float>(subsequentMask(size: ys.shape[1]))
        let target = LangMotionBatch.Target(motion: ys, mask: targetMask)

        // decode motion
        let out = model.decode(sourceMask: source.mask, target: target, memory: memory)
        let singlePreds = model.mixtureModel(out[0...,-1].expandingShape(at: 0))
        
        // perform sampling
        let (sampledMotion, log_probs, done) = MotionDecoder.performNormalMixtureSampling(
            preds: singlePreds, nb_joints: nbJoints, nb_mixtures: nbMixtures, maxMotionLength: maxMotionLength)
        
        // concatenate motion
        ys = Tensor(concatenating: [ys, sampledMotion.expandingShape(at: 0)], alongAxis: 1)        
    }

    // descale motion    
    let descaled_motion = dataset.scaler.inverse_transform(ys.squeezingShape(at:0))
    print("  descaled_motion.shape: \(descaled_motion.shape)")

    let imageURL = dataURL.appendingPathComponent("motion_images/\(prefix).png")
    motionToImg(url: imageURL, motion: descaled_motion, motionFlag: nil, padTo: maxMotionLength, descr: "\(prefix), \(sentence)")
    print("Saved image: \(imageURL.path)")

    let jointNames = dataset.trainExamples[0].motionSample.jointNames
    let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaled_motion)
    let mmmURL = dataURL.appendingPathComponent("motion_images/\(prefix).mmm.xml")
    try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
    print("Saved motion: \(mmmURL.path)")

    if showMotion {
        motionToImg(url: nil, motion: descaled_motion, motionFlag: nil, padTo: maxMotionLength, descr: "\(prefix), \(sentence)")
    }
}

// greedyDecodeMotion(sentence: "human is walking", prefix: "foo9")
// exit(0)

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
let print_every = 10
let limit_print_to_step = 5
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
            if (trainingStepCount < limit_print_to_step || trainingStepCount % print_every == 0) {
                print("==> step \(trainingStepCount)")
            }
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
            let valBatchSize = batch.target.motion.shape[0]

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
        greedyDecodeMotion(sentence: "human is walking", prefix: "epoch_\(epoch+1)")
    }
    summaryWriter.flush()
}

print("\nFinished training.")
