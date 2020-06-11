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

let batchSize = 10
let maxSequenceLength =  300 //600
let runName = "run_8"
let nEpochs = 10
let learningRate: Float = 1e-5

print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("runName: \(runName)")

// let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset_v2.normalized.plist")
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.balanced.515.plist")
let labelsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

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

// instantiate ResNet
var hiddenLayerCount: Int = 12 //12
var attentionHeadCount: Int = 12 //12
var hiddenSize = 64*attentionHeadCount // 64*12 = 768 // 32*6=192 // 64*6=384
let classCount = 5
var featureExtractor = ResNet(classCount: hiddenSize, depth: .resNet18, downsamplingInFirstStage: false, channelCount: 1)

// instantiate FeatureTransformerEncoder
var caseSensitive: Bool = false
var subDirectory: String = "uncased_L-12_H-768_A-12"
let directory = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let vocabularyURL = directory
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


// get a batch
// print("\nOne batch (LabeledMotionBatch):")
// var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
// let epoch = epochIterator.next()
// let batches = Array(epoch!.1)
// let batch = batches[0]
// print("type: \(type(of:batch))")
// print("\nOne motionBatch")
// let motionBatch = batch.data
// print("type: \(type(of:motionBatch))")
// print("motionFrames.shape: \(motionBatch.motionFrames.shape)")
// print("motionFlag.shape: \(motionBatch.motionFlag.shape)")


// run one batch
// print("\nRun one batch:")
// print("==============")
// let classifierOutput = motionClassifier(motionBatch)
// print("classifierOutput.shape: \(classifierOutput.shape)")

// train

// var optimizer = WeightDecayedAdam(
//     for: motionClassifier,
//     learningRate: LinearlyDecayedParameter(
//         baseParameter: LinearlyWarmedUpParameter(
//             baseParameter: FixedParameter<Float>(2e-5),
//             warmUpStepCount: 10,
//             warmUpOffset: 0),
//         // slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
//         slope: -1e-7,  // The LR decays linearly to zero in ~500 steps.
//         startStep: 10),
//     weightDecayRate: 0.01,
//     maxGradientGlobalNorm: 1)

let optimizer = SGD(for: motionClassifier, learningRate: learningRate)
let summaryWriter = SummaryWriter(logdir: URL(fileURLWithPath: "/notebooks/language2motion.gt/data/tboard/").appendingPathComponent(runName), flushMillis: 30*1000)

print("\nTraining MotionClassifier for the motion2Label task!")
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
            // let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Int32>(batch.label))
            // let documents = eagerDocuments.copyingTensorsToDevice(to: device)
            // let labels = Tensor(copying: eagerLabels, to: device)
            let (loss, gradients) = valueWithGradient(at: motionClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            trainingStepCount += 1
            optimizer.update(&motionClassifier, along: gradients)
            // LazyTensorBarrier()
            summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: trainingLossSum / Float(trainingBatchCount))
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
            // let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Int32>(batch.label))
            // let documents = eagerDocuments.copyingTensorsToDevice(to: device)
            // let labels = Tensor(copying: eagerLabels, to: device)

            let logits = motionClassifier(documents)
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
            // LazyTensorBarrier()
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
