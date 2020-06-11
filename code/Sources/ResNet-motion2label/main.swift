import Foundation
import TensorFlow
import PythonKit

import Batcher
import ModelSupport
import Datasets
import ImageClassificationModels
import SummaryWriter
import PythonKit

let batchSize = 10
let maxSequenceLength =  224
let runName = "run_4"
let nEpochs = 3
let learningRate: Float = 0.0001

let metrics = Python.import("sklearn.metrics")

// instantiate ResNet
var model = ResNet(classCount: 5, depth: .resNet18, downsamplingInFirstStage: false, channelCount: 1)
let optimizer = SGD(for: model, learningRate: learningRate)

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.balanced.515.plist")
let labelsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

print("\nLoading dataset...")
let dataset = Motion2Label(
    batchSize: batchSize, 
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    tensorWidth: maxSequenceLength
)
print("dataset.training.count: \(dataset.training.count)")
print("dataset.test.count: \(dataset.test.count)")

let logdirURL = dataURL
                .appendingPathComponent("tboard/ResNet-motion2label/\(runName)", isDirectory: true)
                // .appendingPathComponent(runName, isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

public struct Prediction {
    public let classIdx: Int
    public let className: String
    public let probability: Float
}

extension ResNet {

    public func predict(motionSamples: [MotionSample], labels: [String], batchSize: Int = 10) -> [Prediction] {

        let tensorPairs: Motion2Label.SourceDataSet = motionSamples.map {
            Motion2Label.getTensorPair($0, labelsDict: dataset.labelsDict, labels: dataset.labels, tensorWidth: maxSequenceLength)
        }


        let predBatcher = Batcher(
            on: tensorPairs,
            batchSize: batchSize,
            numWorkers: 1,
            shuffle: false)

        var preds: [Prediction] = []
        Context.local.learningPhase = .inference
        for batch in predBatcher.sequenced() {
            // print("batch")
            let logits = model(batch.first)
            let probs = softmax(logits, alongAxis: 1)
            let classIdxs = logits.argmax(squeezingAxis: 1)
            let batchPreds = (0..<classIdxs.shape[0]).map { 
                (idx) -> Prediction in
                let classIdx: Int = Int(classIdxs[idx].scalar!)
                let prob = probs[idx, classIdx].scalar!
                return Prediction(classIdx: classIdx, className: dataset.labels[classIdx], probability: prob)
            }
            preds.append(contentsOf: batchPreds)
        }

        return preds
    }

}

print("\nStarting ResNet-motion2label training...")
var trainingStepCount = 0
time() {
    for epoch in 1...nEpochs {
        // print("epoch \(epoch)")
        Context.local.learningPhase = .training
        dataset.newTrainCrops()
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        for batch in dataset.training.sequenced() {
            print("progress \(100.0*Float(trainingBatchCount)/Float(dataset.training.count))%")
            let (tensors, labels) = (batch.first, batch.second)
            let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(tensors)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }
            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&model, along: gradients)
            summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: trainingLossSum / Float(trainingBatchCount))
            trainingStepCount += 1
        }
        summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: epoch, value: trainingLossSum / Float(trainingBatchCount))

        Context.local.learningPhase = .inference
        var testLossSum: Float = 0
        var testBatchCount = 0
        var correctGuessCount = 0
        var totalGuessCount = 0
        for batch in dataset.test.sequenced() {
            // print("batch")
            let (tensors, labels) = (batch.first, batch.second)
            let logits = model(tensors)
            testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
            testBatchCount += 1

            let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
            correctGuessCount = correctGuessCount
                + Int(
                    Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount = totalGuessCount + batchSize
        }

        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            [Epoch \(epoch)] \
            Training loss: \(trainingLossSum  / Float(trainingBatchCount)) \
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy*100)%) \
            Loss: \(testLossSum / Float(testBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch, value: testLossSum / Float(testBatchCount))
        summaryWriter.writeScalarSummary(tag: "EpochTestAccuracy", step: epoch, value: accuracy)

        let preds = model.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize)
        let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
        let y_pred = preds.map { $0.className }
        print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
    }
    summaryWriter.flush()

    print("\nFinal stats:")
    print(dataset.labels)
    print()

    let preds = model.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize)

    let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
    let y_pred = preds.map { $0.className }
    print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
}

print("\nFinito.")
