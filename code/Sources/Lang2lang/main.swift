// + implement training
// + implement validation
// + implement inference/decoding
// TODO: use X10
// TODO: train with label texts as target

import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter


let runName = "run_8"
let batchSize = 4000
let maxSequenceLength =  50
let nEpochs = 1
let learningRate: Float = 2e-5

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

// X10 warmup
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
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 256
let feedForwardSize: Int = 1024
let headCount: Int = 8
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
print(device)
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

print("Dataset acquired.")

// get example
// print("example: \(dataset.trainExamples[0])")

// get a batch
// print("\nOne batch (TranslationBatch):")
// var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
// let epoch = epochIterator.next()
// let batches = Array(epoch!.1)
// let batch = batches[0]
// print("type: \(type(of:batch))")
// print("tokenIds.shape: \(batch.tokenIds.shape)")
// print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")

// print()

// run one batch
// print("\nRun one batch:")
// print("==============")
// let output = model(batch)
// print("output.shape: \(output.shape)")

var optimizer = Adam(for: model, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("tboard/Lang2lang/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

@differentiable(wrt: logits)
public func softmaxCrossEntropy2(logits: Tensor<Float>, labels: Tensor<Int32>, ignoreIndex: Int32) -> Tensor<Float> {
    // FIXME: use logits.device, move code back to Utilities.swift
    let ids = Tensor<Int32>(rangeFrom: 0, to: Int32(labels.shape.first!), stride: 1, on: device)
    let indices = ids.gathering(where: labels .!= Tensor(ignoreIndex, on: device))
    let maskedLogits = logits.gathering(atIndices: indices, alongAxis: 0)
    let maskedTargets = labels.gathering(atIndices: indices, alongAxis: 0)
    return softmaxCrossEntropy(logits: maskedLogits, labels: maskedTargets)
}

func update(model: inout TransformerModel, using optimizer: inout Adam<TransformerModel>, for batch: TranslationBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let padIndex = textProcessor.padId
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            softmaxCrossEntropy2(logits: $0.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex)
        }
        optimizer.update(&model, along: grad)
        return loss.scalarized()
    }
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
    return result
}

print("\nTraining Transformer for the Lang2lang task!")
var trainingStepCount = 0
time() {
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(nEpochs).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        for eagerBatch in epochBatches {
            let batch = TranslationBatch(copying: eagerBatch, to: device)
            let loss = update(model: &model, using: &optimizer, for: batch)
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
            let batch = TranslationBatch(copying: eagerBatch, to: device)
            let loss = validate(model: &model, for: batch)
            let valBatchSize = batch.tokenIds.shape[0]

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
    }
    summaryWriter.flush()
}

func greedyDecode(model: TransformerModel, input: TranslationBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input)
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    for _ in 0..<maxLength {
        let decoderInput = TranslationBatch(tokenIds: input.tokenIds,
                                     targetTokenIds: ys,
                                     mask: input.mask,
                                     targetMask: Tensor<Float>(subsequentMask(size: ys.shape[1])),
                                     targetTruth: input.targetTruth,
                                     tokenCount: input.tokenCount)
        let out = model.decode(input: decoderInput, memory: memory)
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1)
    }
    return ys
}

// encode/decode one example
print("\nEncoding/decoding one example")
func decode(tensor: Tensor<Float>, vocab: Vocabulary) -> String {
   var words = [String]()
   for scalar in tensor.scalars {
       if Int(scalar) == endId {
           break
       } else if let token = vocab.token(forId: Int(scalar)) {
           words.append(token)
       }
   }
   return words.joined(separator: " ")
}

var epochIterator2 = dataset.trainingEpochs.enumerated().makeIterator()
let epoch2 = epochIterator2.next()
let batches2 = Array(epoch2!.1)
let batch2 = batches2[0]

let exampleIndex = 1
let source = TranslationBatch(tokenIds: batch2.tokenIds[exampleIndex].expandingShape(at: 0),
                      targetTokenIds: batch2.targetTokenIds[exampleIndex].expandingShape(at: 0),
                      mask: batch2.mask[exampleIndex].expandingShape(at: 0),
                      targetMask: batch2.targetMask[exampleIndex].expandingShape(at: 0),
                      targetTruth: batch2.targetTruth[exampleIndex].expandingShape(at: 0),
                      tokenCount: batch2.tokenCount)
let startId = textProcessor.bosId
let endId = textProcessor.eosId

Context.local.learningPhase = .inference

var outputStr = decode(tensor: source.tokenIds, vocab: textProcessor.vocabulary)
print("source, outputStr: \(outputStr)")

let out = greedyDecode(model: model, input: source, maxLength: 50, startSymbol: startId)

outputStr = decode(tensor: out, vocab: textProcessor.vocabulary)
print("greedyDecode(), outputStr: \(outputStr)")

print("\nFinito.")
