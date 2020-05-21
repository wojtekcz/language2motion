import Foundation
import TensorFlow
import PythonKit

import Batcher
import ModelSupport
import Datasets
import ImageClassificationModels

var model = ResNet(classCount: 5, depth: .resNet18, downsamplingInFirstStage: false, channelCount: 1)
let optimizer = SGD(for: model, learningRate: 0.001)

let batchSize = 25

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset_v1.plist")
let labelsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

let dataset = Motion2Label(
    batchSize: batchSize, 
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL
)
print("dataset.training.count: \(dataset.training.count)")
print("dataset.test.count: \(dataset.test.count)")

print("Starting motion2label training...")

for epoch in 1...10 {
//     print("epoch \(epoch)")
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    for batch in dataset.training.sequenced() {
//         print("progress \(100.0*Float(trainingBatchCount)/Float(dataset.training.count))%")
        let (tensors, labels) = (batch.first, batch.second)
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let logits = model(tensors)
            return softmaxCrossEntropy(logits: logits, labels: labels)
        }
        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        optimizer.update(&model, along: gradients)
    }

    Context.local.learningPhase = .inference
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in dataset.test.sequenced() {
//         print("batch")
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
}
