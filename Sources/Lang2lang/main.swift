// + implement training
// + implement validation
// + implement inference/decoding
// + train with label texts as target
// + use X10

import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter


let runName = "run_5"
let batchSize = 50
// let batchSize = 25
let maxSequenceLength =  50
let nEpochs = 150
// let learningRate: Float = 2e-5
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")
// let dsURL = dataURL.appendingPathComponent("labels_ds_v2.balanced.515.csv")

// X10 warmup
let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
let eagerTensorSum = eagerTensor1 + eagerTensor2
// print(eagerTensorSum)
// print(eagerTensor1.device)
let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
// print(x10Tensor2.device)

// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer, maxSequenceLength: maxSequenceLength)

// instantiate model
let sourceVocabSize = vocabulary.count
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 128
let feedForwardSize: Int = 512
let headCount: Int = 4
let dropoutProbability: Double = 0.1
var model = TransformerModel(
    sourceVocabSize: sourceVocabSize, 
    targetVocabSize: targetVocabSize,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

let device = Device.defaultXLA
// let device = Device.defaultTFEager
print("backend: \(device)")
model.move(to: device)

// load dataset
print("\nLoading dataset...")

var dataset = try Lang2Lang(
    datasetURL: dsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { (example: Lang2Lang.Example) -> TranslationBatch in    
    let singleBatch = textProcessor.preprocess(example: example)
    return singleBatch
}

print("records: \(dataset.trainExamples.count)/\(dataset.valExamples.count)")
print("Dataset acquired.")

func printBatch(_ batch: TranslationBatch) {
    print("type: \(type(of:batch))")

    print("source")
    print("  tokenIds.shape: \(batch.tokenIds.shape)")
    print("  mask.shape: \(batch.mask.shape)")

    print("target")
    print("  targetTokenIds.shape: \(batch.targetTokenIds.shape)")
    print("  targetMask.shape: \(batch.targetMask.shape)")
    print("  targetTruth.shape: \(batch.targetTruth.shape)")
    print("  tokenCount: \(batch.tokenCount)")
}

/// one example to single batch
//print("\nSingle batch")
//print("============")
//let example = dataset.trainExamples[0]
//print("example: \(example)")
//print("example.sourceSentence: \"\(example.sourceSentence)\"")

//let singleBatch = textProcessor.preprocess(example: example)
//printBatch(singleBatch)

// get a batch
//print("\nOne batch:")
//var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
//let epoch = epochIterator.next()
//let batches = Array(epoch!.1)
//let batch: TranslationBatch = batches[0]
//printBatch(batch)

// print("targetTokenIds: \(batch.targetTokenIds)")
// print("targetTruth: \(batch.targetTruth)")
// print("targetMask: \(batch.targetMask)")

// run one batch
//print("\nRun one batch:")
//print("==============")
//let output = model(batch)
//print("output.shape: \(output.shape)")

var optimizer = Adam(for: model, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("runs/Lang2lang/\(runName)", isDirectory: true)
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

func update(model: inout TransformerModel, using optimizer: inout Adam<TransformerModel>, for batch: TranslationBatch) -> Float {
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
        // print("  LazyTensorBarrier()")
        // time {
            LazyTensorBarrier()
        // }
        return loss.scalarized()
    }
    // print("update() - stop")
    return result
}

/// returns validation loss
func validate(model: inout TransformerModel, for batch: TranslationBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let padIndex = textProcessor.padId
    let result = withLearningPhase(.inference) { () -> Float in
        softmaxCrossEntropy2(logits: model.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex).scalarized()
    }
    // print("LazyTensorBarrier() - validate")
    // time {
        LazyTensorBarrier()
    // }
    return result
}

// setup decoding
func greedyDecode(model: TransformerModel, input: TranslationBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input).lastLayerOutput
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    for _ in 0..<maxLength {

        let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
        var motionPartMask = TranslationBatch.makeStandardMask(target: motionPartFlag, pad: 0, shiftRight: true)
        let motionLen = Int(motionPartFlag.sum().scalar!)
        motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
        motionPartMask = abs(motionPartMask)

        let decoderInput = TranslationBatch(tokenIds: input.tokenIds,
                                     targetTokenIds: ys,
                                     mask: input.mask,
                                     targetMask: motionPartMask,
                                     targetTruth: input.targetTruth,
                                     tokenCount: input.tokenCount)

        let out = model.decode(input: decoderInput, memory: memory)
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1) // , on: device
    }
    return ys
}

// encode/decode one example
func greedyDecodeFromString(inputStr: String, maxLength: Int = 50, model: TransformerModel) -> String {
    let example = Lang2Lang.Example(id: "-1", sourceSentence: inputStr, targetSentence: "")
    let singleBatch = textProcessor.preprocess(example: example)
    let decodedTensor = greedyDecode(model: model, input: singleBatch, maxLength: maxLength, startSymbol: textProcessor.bosId)
    return decode(tensor: decodedTensor, vocab: textProcessor.vocabulary)
}

let textsToDecode = [
    "A person runs forward.",
    "A human is swimming.",
    "A person walks.",
    "A person plays the air guitar",
    "A person performs a squat.",
    "A human raises their left foot and touches it with the right hand."
]

print("\nTraining Transformer for the Lang2lang task!")
var trainingStepCount = 0
time() {
    // print("LazyTensorBarrier() - before training loops")
    LazyTensorBarrier()
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(nEpochs).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        time() {
            for eagerBatch in epochBatches {
                let batch = TranslationBatch(copying: eagerBatch, to: device)
                let loss: Float = update(model: &model, using: &optimizer, for: batch)
                // print("current loss at step \(trainingStepCount): \(loss)")
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
                let batch = TranslationBatch(copying: eagerBatch, to: device)
                let loss: Float = validate(model: &model, for: batch)
                let valBatchSize = batch.tokenIds.shape[0]

                devLossSum += loss
                devBatchCount += 1
                totalGuessCount += valBatchSize
            }

            print(
                """
                Eval loss: \(devLossSum / Float(devBatchCount))
                """
            )
            summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch+1, value: devLossSum / Float(devBatchCount))

            print("\nDecoding few texts:") // on eager device
            Context.local.learningPhase = .inference
            model.move(to: Device.defaultTFEager)
            for text in textsToDecode {
                let decodedText = greedyDecodeFromString(inputStr: text, maxLength: 15, model: model)
                print("\"\(text)\" -> \"\(decodedText)\"")
            }
            model.move(to: device)
        }
    }
    summaryWriter.flush()
}

print("\nFinito.")
