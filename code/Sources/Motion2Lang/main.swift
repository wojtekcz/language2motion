import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionModels


let runName = "run_1"
// let batchSize = 4000
let batchSize = 1000
// let batchSize = 200
let maxSequenceLength =  50
let nEpochs = 2
// let learningRate: Float = 2e-5
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.plist")
// let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.normalized.downsampled.sampled.490.plist")
let langDatasetURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

// X10 warmup
// let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
// let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
// let eagerTensorSum = eagerTensor1 + eagerTensor2
// print(eagerTensorSum)
// print(eagerTensor1.device)
// let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
// print(x10Tensor2.device)

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

// let device = Device.defaultXLA
// print(device)
// model.move(to: device)

// load dataset
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

// get example
let example = dataset.trainExamples[0]
print("example.id: \(example.id)")
print("example.motionSample.timestepsArray.last: \(example.motionSample.timestepsArray.last!)")
print("example.motionSample.motionFramesArray.shape: \(example.motionSample.motionFramesArray.shape)")
print("example.targetSentence: \(example.targetSentence)")

// get a batch
print("\nOne batch (MotionLangBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch: MotionLangBatch = batches[0]
print("type: \(type(of:batch))")
print("motionFrames.shape: \(batch.motionFrames.shape)")
// print("motionFlag.shape: \(batch.motionFlag.shape)")
print("mask.shape: \(batch.mask.shape)")
print("origMotionFramesCount.shape: \(batch.origMotionFramesCount.shape)")
print("origMotionFramesCount: \(batch.origMotionFramesCount)")
print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")
print("targetMask.shape: \(batch.targetMask.shape)")
print("targetTruth.shape: \(batch.targetTruth.shape)")

print()

// run one batch
print("\nRun one batch:")
print("==============")
let output = model(batch)
print("output.shape: \(output.shape)")

var optimizer = Adam(for: model, learningRate: learningRate)
// optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("tboard/Motion2lang/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

@differentiable(wrt: logits)
public func softmaxCrossEntropy2(logits: Tensor<Float>, labels: Tensor<Int32>, ignoreIndex: Int32) -> Tensor<Float> {
    // print("softmaxCrossEntropy2() - start")
    // FIXME: use logits.device, move code back to Utilities.swift
    // print("  LazyTensorBarrier()")
    // time {
    //     LazyTensorBarrier()
    // }
    // let ids = Tensor<Int32>(rangeFrom: 0, to: Int32(labels.shape.first!), stride: 1, on: device)    
    // let indices = ids.gathering(where: labels .!= Tensor(ignoreIndex, on: device))
    // let maskedLogits = logits.gathering(atIndices: indices, alongAxis: 0)
    // let maskedTargets = labels.gathering(atIndices: indices, alongAxis: 0)
    // print("  maskedLogits.shape: \(maskedLogits.shape)")
    // print("  maskedTargets.shape: \(maskedTargets.shape)")
    let sce = softmaxCrossEntropy(logits: logits, labels: labels)
    // print("sce: \(sce)")
    // let maskedSCE = softmaxCrossEntropy(logits: maskedLogits, labels: maskedTargets)
    // print("maskedSCE: \(maskedSCE)")
    // print("softmaxCrossEntropy2() - stop")
    return sce
}

func update(model: inout MotionLangTransformer, using optimizer: inout Adam<MotionLangTransformer>, for batch: MotionLangBatch) -> Float {
    // print("update() - start")
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    // print("  resultSize: \(resultSize)")
    let padIndex = textProcessor.padId
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            (model) -> Tensor<Float> in
            let logits = model.generate(input: batch).reshaped(to: [resultSize, -1])
            // print("  logits.shape: \(logits.shape)")
            // print("  labels.shape: \(labels.shape)")
            let sce = softmaxCrossEntropy2(logits: logits, labels: labels,ignoreIndex: padIndex)
            return sce
        }
        optimizer.update(&model, along: grad)
        LazyTensorBarrier()
        return loss.scalarized()
    }
    // print("update() - stop")
    return result
}

/// returns validation loss
func validate(model: inout MotionLangTransformer, for batch: MotionLangBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let padIndex = textProcessor.padId
    let result = withLearningPhase(.inference) { () -> Float in
        softmaxCrossEntropy2(logits: model.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex).scalarized()
    }
    LazyTensorBarrier()
    return result
}

// setup decoding
var epochIterator2 = dataset.trainingEpochs.enumerated().makeIterator()
let epoch2 = epochIterator2.next()
let batches2 = Array(epoch2!.1)
let batch2 = batches2[0]

let exampleIndex = 1 // FIXME: utilize exampleIndex
var source = batch2 //Motion2Lang.reduceDataBatches(batches2)

print()

func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input)
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    // ys = Tensor(copying: ys, to: device)
    for _ in 0..<maxLength {
        let decoderInput = MotionLangBatch(motionFrames: input.motionFrames,
                                     mask: input.mask,
                                     origMotionFramesCount: input.origMotionFramesCount,
                                     targetTokenIds: ys,
                                     targetMask: Tensor<Float>(subsequentMask(size: ys.shape[1])),
                                     targetTruth: input.targetTruth)
        // decoderInput = MotionLangBatch(copying: decoderInput, to: device)
        let out = model.decode(input: decoderInput, memory: memory)
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1) // , on: device
        // ys = Tensor(copying: ys, to: device)
    }
    return ys
}

func decode(tensor: Tensor<Float>, vocab: Vocabulary) -> String {
   var words = [String]()
   for scalar in tensor.scalars {
       if Int(scalar) == textProcessor.eosId {
           break
       } else if let token = vocab.token(forId: Int(scalar)) {
           words.append(token)
       }
   }
   return words.joined(separator: " ")
}

var outputStr = decode(tensor: source.targetTokenIds, vocab: textProcessor.vocabulary)
print("decode(source.targetTokenIds): \(outputStr)")

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
            print("==> step \(trainingStepCount)")
            // print("eagerBatch.tokenIds.shape: \(eagerBatch.tokenIds.shape)")
            // print("eagerBatch.targetTokenIds.shape: \(eagerBatch.targetTokenIds.shape)")
            // print("eagerBatch.mask.shape: \(eagerBatch.mask.shape)")
            // print("eagerBatch.targetTruth.shape: \(eagerBatch.targetTruth.shape)")
            // print("eagerBatch.tokenCount: \(eagerBatch.tokenCount)")
            let batch = eagerBatch //MotionLangBatch(copying: eagerBatch, to: device)
            let loss: Float = update(model: &model, using: &optimizer, for: batch)
            print("current loss at step \(trainingStepCount): \(loss)")
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
            let batch = eagerBatch //MotionLangBatch(copying: eagerBatch, to: device)
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

        print("\nEncoding/decoding one example") // on eager device
        Context.local.learningPhase = .inference
        source = MotionLangBatch(copying: source, to: Device.defaultTFEager)
        // model.move(to: Device.defaultTFEager)
        // let out = greedyDecode(model: model, input: source, maxLength: 50, startSymbol: textProcessor.bosId)
        // outputStr = decode(tensor: out, vocab: textProcessor.vocabulary)
        // print("greedyDecode(): \"\(outputStr)\"")
        // model.move(to: device)
    }
    summaryWriter.flush()
}

// encode/decode one example
// print("\nEncoding/decoding one example")
// Context.local.learningPhase = .inference
// source = MotionLangBatch(copying: source, to: device)
// model.move(to: Device.defaultTFEager)
// let out = greedyDecode(model: model, input: source, maxLength: 50, startSymbol: textProcessor.bosId)
// outputStr = decode(tensor: out, vocab: textProcessor.vocabulary)
// print("greedyDecode(), outputStr: \(outputStr)")

print("\nFinito.")
