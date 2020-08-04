import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionLangModels

/// Set training params
let runName = "run_1"
let batchSize = 6000
// let batchSize = 3000
let maxSequenceLength =  10
let nEpochs = 1
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.mini.plist")
let langDatasetURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

/// Select eager or X10 backend

// let device = Device.defaultXLA
let device = Device.defaultTFEager
print(device)


/// X10 warm-up
let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
let eagerTensorSum = eagerTensor1 + eagerTensor2
print(eagerTensorSum)
print(eagerTensor1.device)
let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
print(x10Tensor2.device)

// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer, maxSequenceLength: maxSequenceLength)

// instantiate model
let sourceVocabSize = vocabulary.count
let inputSize = 48 // TODO: get value from dataset
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
let dropoutProbability: Double = 0.1

var model = MotionLangTransformer(
    sourceVocabSize: sourceVocabSize, 
    inputSize: inputSize,
    targetVocabSize: targetVocabSize,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

model.move(to: device)

/// load dataset
print("\nLoading dataset...")

var dataset = try Motion2Lang(
    motionDatasetURL: motionDatasetURL,
    langDatasetURL: langDatasetURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { (example: Motion2Lang.Example) -> MotionLangBatch in    
    let singleBatch = textProcessor.preprocess(example: example)
    return singleBatch
}

print("Dataset acquired.")

/// Test model with one batch
// get a batch
// print("\nOne batch (MotionLangBatch):")
// var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
// let epoch = epochIterator.next()
// let batches = Array(epoch!.1)
// let batch: MotionLangBatch = batches[0]
// print("type: \(type(of:batch))")
// print("motionFrames.shape: \(batch.motionFrames.shape)")
// // print("motionFlag.shape: \(batch.motionFlag.shape)")
// print("mask.shape: \(batch.mask.shape)")
// print("origMotionFramesCount.shape: \(batch.origMotionFramesCount.shape)")
// print("origMotionFramesCount: \(batch.origMotionFramesCount)")
// print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")
// print("targetMask.shape: \(batch.targetMask.shape)")
// print("targetTruth.shape: \(batch.targetTruth.shape)")

// run one batch
// print("\nRun one batch:")
// print("==============")
// let deviceBatch = MotionLangBatch(copying: batch, to: device)
// let output = model(deviceBatch)
// print("output.shape: \(output.shape)")

/// Optimizer
var optimizer = Adam(for: model, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("tboard/Motion2lang/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

/// Training helpers
func update(model: inout MotionLangTransformer, using optimizer: inout Adam<MotionLangTransformer>, for batch: MotionLangBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            (model) -> Tensor<Float> in
            let logits = model.generate(input: batch).reshaped(to: [resultSize, -1])
            let sce = softmaxCrossEntropy(logits: logits, labels: labels)
            return sce
        }
        optimizer.update(&model, along: grad)
        LazyTensorBarrier()
        return loss.scalarized()
    }
    return result
}

/// returns validation loss
func validate(model: inout MotionLangTransformer, for batch: MotionLangBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let result = withLearningPhase(.inference) { () -> Float in
        softmaxCrossEntropy(logits: model.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels).scalarized()
    }
    LazyTensorBarrier()
    return result
}

/// Set up decoding

func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input)
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    // ys = Tensor(copying: ys, to: device)
    for i in 0..<maxLength {
        print("loop \(i)")
        print("  ys: \(ys)")
        let decoderInput = MotionLangBatch(motionFrames: input.motionFrames,
                                     mask: input.mask,
                                     origMotionFramesCount: input.origMotionFramesCount,
                                     targetTokenIds: ys,
                                     targetMask: Tensor<Float>(subsequentMask(size: ys.shape[1])),
                                     targetTruth: input.targetTruth)
        // decoderInput = MotionLangBatch(copying: decoderInput, to: device)
        let out = model.decode(input: decoderInput, memory: memory)
        print("out.shape: \(out.shape)")
        print("out[0...,-1].shape: \(out[0...,-1].shape)")
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1) // , on: device
        // ys = Tensor(copying: ys, to: device)
    }
    return ys
}

func greedyDecodeSample(_ sample_id: Int) {
    // get example
    let ms = dataset.motionSampleDict[sample_id]!
    let langRec = dataset.langRecsDict[sample_id]!
    let example = Motion2Lang.getExample(motionSample: ms, langRec: langRec)
    print("example.id: \(example.id)")
    print("example.motionSample.timestepsArray.last: \(example.motionSample.timestepsArray.last!)")
    print("example.motionSample.motionFramesArray.shape: \(example.motionSample.motionFramesArray.shape)")
    print("example.targetSentence: \(example.targetSentence)")

    let singleExampleBatch = textProcessor.preprocess(example: example)
    var source = Motion2Lang.reduceDataBatches([singleExampleBatch])
    print("\nEncoding/decoding one example") // on eager device
    Context.local.learningPhase = .inference
    source = MotionLangBatch(copying: source, to: Device.defaultTFEager)
    model.move(to: Device.defaultTFEager)
    let out = greedyDecode(model: model, input: source, maxLength: maxSequenceLength, startSymbol: textProcessor.bosId)
    let outputStr = textProcessor.decode(tensor: out)
    print("greedyDecode(): \"\(outputStr)\"")
    model.move(to: device)
}

// get example
let example = dataset.trainExamples[0]
greedyDecodeSample(Int(example.id)!)

/// Training loop
print("\nTraining Transformer for the Motion2lang task!")
var trainingStepCount = 0
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
            if (trainingStepCount < 5) {
                print("==> step \(trainingStepCount)")
            }
            let batch = MotionLangBatch(copying: eagerBatch, to: device)
            let loss: Float = update(model: &model, using: &optimizer, for: batch)
            if (trainingStepCount < 5) {
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
            let batch = MotionLangBatch(copying: eagerBatch, to: device)
            let loss: Float = validate(model: &model, for: batch)
            let valBatchSize = batch.motionFrames.shape[0]

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
        greedyDecodeSample(Int(example.id)!)
    }
    summaryWriter.flush()
}

print("\nFinished training.")

/// Generate motion description

let sample_id = 446
greedyDecodeSample(sample_id)

print("\nFinito.")
