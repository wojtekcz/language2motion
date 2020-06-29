// TODO: + implement training
// TODO: implement validation
// TODO: implement inference/decoding
// TODO: use X10

import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter


let runName = "run_2"
let batchSize = 4000
let maxSequenceLength =  50
let nEpochs = 20
let learningRate: Float = 2e-5

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

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
print("example: \(dataset.trainExamples[0])")

// get a batch
print("\nOne batch (TranslationBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch = batches[0]
print("type: \(type(of:batch))")
print("tokenIds.shape: \(batch.tokenIds.shape)")
print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")

print()

// run one batch
print("\nRun one batch:")
print("==============")
let output = model(batch)
print("output.shape: \(output.shape)")

// TODO: * implement training

var optimizer = Adam(for: model, learningRate: learningRate)

let logdirURL = dataURL.appendingPathComponent("tboard/Lang2lang/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

func update(model: inout TransformerModel, using optimizer: inout Adam<TransformerModel>, for batch: TranslationBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let padIndex = textProcessor.padId
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            softmaxCrossEntropy(logits: $0.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels,ignoreIndex: padIndex)
        }
        optimizer.update(&model, along: grad)
        return loss.scalarized()
    }
    return result
}

print("Training Transformer for the Lang2lang task!")
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

        for batch in epochBatches {
            // let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            // let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            //     let logits = model(documents)
            //     return softmaxCrossEntropy(logits: logits, labels: labels)
            // }

            let loss = update(model: &model, using: &optimizer, for: batch)
            print("current loss at step \(trainingStepCount): \(loss)")

            trainingLossSum += loss//.scalarized()
            trainingBatchCount += 1
            // optimizer.update(&model, along: gradients)
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
        // Context.local.learningPhase = .inference
        // var devLossSum: Float = 0
        // var devBatchCount = 0
        // var correctGuessCount = 0
        // var totalGuessCount = 0

        // for batch in dataset.validationBatches {
        //     let valBatchSize = batch.data.tokenIds.shape[0]

        //     let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
        //     let logits = model(documents)
        //     let loss = softmaxCrossEntropy(logits: logits, labels: labels)
        //     devLossSum += loss.scalarized()
        //     devBatchCount += 1

        //     let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels

        //     correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        //     totalGuessCount += valBatchSize
        // }
        
        // let testAccuracy = Float(correctGuessCount) / Float(totalGuessCount)
        // print(
        //     """
        //     Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(testAccuracy)) \
        //     Eval loss: \(devLossSum / Float(devBatchCount))
        //     """
        // )
        // summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch+1, value: devLossSum / Float(devBatchCount))
        // summaryWriter.writeScalarSummary(tag: "EpochTestAccuracy", step: epoch+1, value: testAccuracy)
    }
    summaryWriter.flush()
}

print("\nFinito.")
