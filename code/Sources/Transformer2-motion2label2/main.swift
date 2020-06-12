import Foundation
import TensorFlow
import Datasets
import MotionModels
import ImageClassificationModels
import TextModels
import ModelSupport
import SummaryWriter
import PythonKit

let metrics = Python.import("sklearn.metrics")

let runName = "run_2"
let batchSize = 10
let maxSequenceLength =  300
let nEpochs = 30
let resNetLearningRate: Float = 1e-3
let learningRate: Float = 2e-5

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("resNetLearningRate: \(resNetLearningRate)")
print("learningRate: \(learningRate)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.balanced.515.plist")
let labelsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

//======================================== ResNet =================================
print ("\n===== ResNet training =====")
// instantiate ResNet
let classCount = 5
var featureExtractor = ResNet(classCount: classCount, depth: .resNet18, downsamplingInFirstStage: true, channelCount: 1)
let resNetOptimizer = SGD(for: featureExtractor, learningRate: resNetLearningRate)

print("\nLoading dataset...")
let resNetDataset = Motion2Label(
    batchSize: batchSize, 
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    tensorWidth: maxSequenceLength
)
print("resNetDataset.training.count: \(resNetDataset.training.count)")
print("resNetDataset.test.count: \(resNetDataset.test.count)")

let resNetLogdirURL = dataURL.appendingPathComponent("tboard/Transformer2-motion2label2/\(runName)-ResNet", isDirectory: true)
let resNetSummaryWriter = SummaryWriter(logdir: resNetLogdirURL, flushMillis: 30*1000)


print("\nStarting ResNet-motion2label training...")
time() {
    var trainingStepCount = 0
    for epoch in 1...nEpochs {
        // print("epoch \(epoch)")
        Context.local.learningPhase = .training
        resNetDataset.newTrainCrops()
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        for batch in resNetDataset.training.sequenced() {
            // print("progress \(100.0*Float(trainingBatchCount)/Float(resNetDataset.training.count))%")
            let (tensors, labels) = (batch.first, batch.second)
            let (loss, gradients) = valueWithGradient(at: featureExtractor) { model -> Tensor<Float> in
                let logits = model(tensors)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }
            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            resNetOptimizer.update(&featureExtractor, along: gradients)
            resNetSummaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: trainingLossSum / Float(trainingBatchCount))
            trainingStepCount += 1
        }
        resNetSummaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: epoch, value: trainingLossSum / Float(trainingBatchCount))

        Context.local.learningPhase = .inference
        var testLossSum: Float = 0
        var testBatchCount = 0
        var correctGuessCount = 0
        var totalGuessCount = 0
        for batch in resNetDataset.test.sequenced() {
            // print("batch")
            let (tensors, labels) = (batch.first, batch.second)
            let logits = featureExtractor(tensors)
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
        resNetSummaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch, value: testLossSum / Float(testBatchCount))
        resNetSummaryWriter.writeScalarSummary(tag: "EpochTestAccuracy", step: epoch, value: accuracy)
    }
    resNetSummaryWriter.flush()

    print()
}

//======================================== Transformer =================================

print ("\n===== Transformer training =====")

print("\nLoading dataset...")
let dataset = try! Motion2Label2(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize
) { 
    // TODO: move this to dataset class
    (example: Motion2LabelExample) -> LabeledMotionBatch in
    let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
    let motionFlag = Tensor<Int32>(motionFrames[0..., 44...44].squeezingShape(at: 1))
    let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))
    let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag, origMotionFramesCount: origMotionFramesCount)
    let label = Tensor<Int32>(Int32(example.label!.idx))
    return LabeledMotionBatch(data: motionBatch, label: label)
}

print("dataset.trainingExamples.count: \(dataset.trainingExamples.count)")
print("dataset.validationExamples.count: \(dataset.validationExamples.count)")

// print("dataset.trainingExamples[0]: \(dataset.trainingExamples[0])")

// instantiate FeatureTransformerEncoder

var hiddenLayerCount: Int = 8 //12
var attentionHeadCount: Int = 8 //12
var hiddenSize = 64*attentionHeadCount

print("hiddenSize: \(hiddenSize)")

var caseSensitive: Bool = false
var subDirectory: String = "uncased_L-12_H-768_A-12"
let vocabularyURL = dataURL
    .appendingPathComponent(subDirectory)
    .appendingPathComponent("vocab.txt")

let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary,
    caseSensitive: caseSensitive, unknownToken: "[UNK]", maxTokenLength: nil)

var variant: BERT.Variant = .bert          
var intermediateSize: Int = hiddenSize*4 // 3072/768=4

var transformerEncoder = FeatureTransformerEncoder(
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

// instantiate MotionClassifier
var motionClassifier = MotionClassifier(featureExtractor: featureExtractor, transformerEncoder: transformerEncoder, classCount: classCount, maxSequenceLength: maxSequenceLength)

let optimizer = SGD(for: motionClassifier, learningRate: learningRate)
let logdirURL = dataURL.appendingPathComponent("tboard/Transformer2-motion2label2/\(runName)-Transformer", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

print("\nTraining MotionClassifier for the motion2Label task!")
time() {
    var trainingStepCount = 0
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
            let (loss, gradients) = valueWithGradient(at: motionClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&motionClassifier, along: gradients)
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
            let valBatchSize = batch.data.motionFrames.shape[0]

            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))

            let logits = motionClassifier(documents)
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

        let preds = motionClassifier.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize)
        let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
        let y_pred = preds.map { $0.className }
        print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
    }
    summaryWriter.flush()
}

print("\nFinal stats:")
print(dataset.labels)
print()
let preds = motionClassifier.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize)
let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
let y_pred = preds.map { $0.className }
print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
print(metrics.classification_report(y_true, y_pred, labels: dataset.labels, zero_division: false))

print("\nFinito.")
