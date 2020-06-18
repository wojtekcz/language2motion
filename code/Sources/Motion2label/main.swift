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
let nEpochs = 10
let learningRate: Float = 2e-5
let logdir = "tboard/Motion2label/\(runName)"

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")
print("logdir: \(logdir)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
// let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.balanced.515.plist")
// let balanceClassSamples: Int? = nil
let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.normalized.downsampled.sampled.490.plist")
let balanceClassSamples: Int? = 60
// let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset.motion_flag.normalized.plist")
// let balanceClassSamples: Int? = 600
let labelsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")


print("\nLoading dataset...")
print(serializedDatasetURL.path)
print(labelsURL.path)
let dataset = try! Motion2Label(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize,
    balanceClassSamples: balanceClassSamples
) { 
    // TODO: move this to dataset class
    (example: Motion2LabelExample) -> LabeledMotionBatch in
    let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
    let mfIdx = MotionFrame.cjpMotionFlagIdx
    let motionFlag = Tensor<Int32>(motionFrames[0..., mfIdx...mfIdx].squeezingShape(at: 1))
    let origMotionFramesCount = Tensor<Int32>(Int32(motionFrames.shape[0]))
    let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag, origMotionFramesCount: origMotionFramesCount)
    let label = Tensor<Int32>(Int32(example.label!.idx))
    return LabeledMotionBatch(data: motionBatch, label: label)
}

print("dataset.trainingExamples.count: \(dataset.trainingExamples.count)")
print("dataset.validationExamples.count: \(dataset.validationExamples.count)")

// print("dataset.trainingExamples[0]: \(dataset.trainingExamples[0])")

// instantiate FeatureTransformerEncoder
var hiddenLayerCount: Int = 4 //12
var attentionHeadCount: Int = 4 //12
var hiddenSize = 96*attentionHeadCount // 64*12 = 768 // 32*6=192 // 64*6=384 // 64*8=512
let classCount = 5

// TODO: make training work with ResNet, too
// var resNetModel = ResNet(classCount: hiddenSize, depth: .resNet18, downsamplingInFirstStage: true, channelCount: 1)

var caseSensitive: Bool = false
let vocabularyURL = dataURL.appendingPathComponent("uncased_L-12_H-768_A-12/vocab.txt")

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
    maxSequenceLength: maxSequenceLength,
    typeVocabularySize: 2,
    initializerStandardDeviation: 0.02,
    useOneHotEmbeddings: false)

print("\nFeatureTransformerEncoder stats:")
print("hiddenLayerCount: \(hiddenLayerCount)")
print("attentionHeadCount: \(attentionHeadCount)")
print("hiddenSize: \(hiddenSize)")


// instantiate DenseMotionClassifier
let inputSize = dataset.motionDataset.motionSamples[0].motionFramesArray.shape[1]
var motionClassifier = DenseMotionClassifier(transformerEncoder: transformerEncoder, inputSize: inputSize, classCount: classCount, maxSequenceLength: maxSequenceLength)


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
//             baseParameter: FixedParameter<Float>(learningRate),
//             warmUpStepCount: 10,
//             warmUpOffset: 0),
//         // slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
//         slope: (-5e-7)/10,  // The LR decays linearly to zero in ~1000? steps.
//         startStep: 10),
//     weightDecayRate: 0.01/2,
//     maxGradientGlobalNorm: 1)

// let optimizer = SGD(for: motionClassifier, learningRate: learningRate)
let optimizer = Adam(for: motionClassifier, learningRate: learningRate)

print("optimizer = \(optimizer)")

let logdirURL = dataURL.appendingPathComponent(logdir, isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

print("\nTraining DenseMotionClassifier for the motion2Label task!")
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
            let (loss, gradients) = valueWithGradient(at: motionClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&motionClassifier, along: gradients)
            summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: trainingLossSum / Float(trainingBatchCount))
            // summaryWriter.writeScalarSummary(tag: "LearningRate", step: trainingStepCount, value: optimizer.lastLearningRate)
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
let preds = motionClassifier.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize)
let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
let y_pred = preds.map { $0.className }
print("accuracy: \(metrics.accuracy_score(y_true, y_pred))")
print(dataset.labels)
print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
print(metrics.classification_report(y_true, y_pred, labels: dataset.labels, zero_division: false))

print("\nFinito.")
