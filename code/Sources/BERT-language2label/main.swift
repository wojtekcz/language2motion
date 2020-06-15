import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels
import SummaryWriter
import PythonKit

let metrics = Python.import("sklearn.metrics")

let runName = "run_3"
let batchSize = 512
let maxSequenceLength =  50
let nEpochs = 20
let learningRate: Float = 2e-5

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

print("\nBERT stats:")
var hiddenLayerCount: Int = 8 //12
var attentionHeadCount: Int = 8 //12
var hiddenSize = 64*attentionHeadCount // 64*12 = 768 // 32*6=192 // 64*6=384
let classCount = 5
print("hiddenLayerCount: \(hiddenLayerCount)")
print("attentionHeadCount: \(attentionHeadCount)")
print("hiddenSize: \(hiddenSize)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let dsURL = dataURL.appendingPathComponent("labels_ds_v2.balanced.515.csv")

// let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
// let workspaceURL = URL(
//     fileURLWithPath: "bert_models", isDirectory: true,
//     relativeTo: URL(
//         fileURLWithPath: NSTemporaryDirectory(),
//         isDirectory: true))
// let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)

// instantiate BERT
var caseSensitive: Bool = false
let vocabularyURL = dataURL.appendingPathComponent("uncased_L-12_H-768_A-12/vocab.txt")

let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary,
    caseSensitive: caseSensitive, unknownToken: "[UNK]", maxTokenLength: nil)

var variant: BERT.Variant = .bert          
var intermediateSize: Int = hiddenSize*4 // 3072/768=4
print("intermediateSize: \(intermediateSize)")

let bert = BERT(
    variant: variant,
    vocabulary: vocabulary,
    tokenizer: tokenizer,
    caseSensitive: caseSensitive,
    hiddenSize: hiddenSize,
    hiddenLayerCount: hiddenLayerCount,
    attentionHeadCount: attentionHeadCount,
    intermediateSize: intermediateSize,
    intermediateActivation: gelu,
    hiddenDropoutProbability: 0.1,
    attentionDropoutProbability: 0.1,
    maxSequenceLength: 512,
    typeVocabularySize: 2,
    initializerStandardDeviation: 0.02,
    useOneHotEmbeddings: false)

var bertClassifier = BERTClassifier(bert: bert, classCount: classCount)

print("\nLoading dataset...")
var dataset = try Language2Label(
    datasetURL: dsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize,
    entropy: SystemRandomNumberGenerator()
) { (example: Language2LabelExample) -> LabeledTextBatch in
    let textBatch = bertClassifier.bert.preprocess(
        sequences: [example.text],
        maxSequenceLength: maxSequenceLength)
   return (data: textBatch, 
           label: example.label.map { 
               (label: Language2LabelExample.LabelTuple) in Tensor(Int32(label.idx))
           }!
          )
}

print("Dataset acquired.")

var optimizer = WeightDecayedAdam(
    for: bertClassifier,
    learningRate: LinearlyDecayedParameter(
        baseParameter: LinearlyWarmedUpParameter(
            baseParameter: FixedParameter<Float>(learningRate),
            warmUpStepCount: 10,
            warmUpOffset: 0),
        // slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
        // slope: -1e-7,  // The LR decays linearly to zero in ~500 steps.
        slope: -0.5e-7,  // The LR decays linearly to zero in ~100 steps.
        startStep: 10),
    weightDecayRate: 0.01,
    maxGradientGlobalNorm: 1)

let logdirURL = dataURL.appendingPathComponent("tboard/BERT-language2label/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

print("Training BERT for the Language2Label task!")
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
            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            let (loss, gradients) = valueWithGradient(at: bertClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&bertClassifier, along: gradients)
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
        var correctGuessCount = 0
        var totalGuessCount = 0

        for batch in dataset.validationBatches {
            let valBatchSize = batch.data.tokenIds.shape[0]

            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            let logits = bertClassifier(documents)
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
            devLossSum += loss.scalarized()
            devBatchCount += 1

            let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels

            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += valBatchSize
        }
        
        let testAccuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(testAccuracy)) \
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch+1, value: devLossSum / Float(devBatchCount))
        summaryWriter.writeScalarSummary(tag: "EpochTestAccuracy", step: epoch+1, value: testAccuracy)

        let testTexts: [String] = dataset.validationSamples.map{$0.text}
        let preds = bertClassifier.predict(testTexts, maxSequenceLength: maxSequenceLength, labels: dataset.labels, batchSize: batchSize)
        let y_true = dataset.validationSamples.map{$0.label}
        let y_pred = preds.map { $0.className }
        print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
    }
    summaryWriter.flush()
}

print("\nFinal stats:")
print(dataset.labels)
print()
let testTexts: [String] = dataset.validationSamples.map{$0.text}
let preds = bertClassifier.predict(testTexts, maxSequenceLength: maxSequenceLength, labels: dataset.labels, batchSize: batchSize)
let y_true = dataset.validationSamples.map{$0.label}
let y_pred = preds.map { $0.className }
print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
print(metrics.classification_report(y_true, y_pred, labels: dataset.labels, zero_division: false))

print("\nFinito.")
