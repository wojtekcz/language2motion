import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels


let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(
    fileURLWithPath: "bert_models", isDirectory: true,
    relativeTo: URL(
        fileURLWithPath: NSTemporaryDirectory(),
        isDirectory: true))
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 5)

let maxSequenceLength = 50
let batchSize = 3096

let dsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

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
            baseParameter: FixedParameter<Float>(2e-5),
            warmUpStepCount: 10,
            warmUpOffset: 0),
        // slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
        slope: -1e-7,  // The LR decays linearly to zero in ~500 steps.
        startStep: 10),
    weightDecayRate: 0.01,
    maxGradientGlobalNorm: 1)

print("Training BERT for the Language2Label task!")

time() {
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(5).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        print("epochBatches.count: \(epochBatches.count)")

        for batch in epochBatches {
            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            let (loss, gradients) = valueWithGradient(at: bertClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&bertClassifier, along: gradients)

            print(
                """
                Training loss: \(trainingLossSum / Float(trainingBatchCount))
                """
            )
        }

        print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
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
        
        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
    }
}
