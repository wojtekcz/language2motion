import Foundation
import TensorFlow
import Datasets
import TextModels
import MotionModels
import ImageClassificationModels
import ModelSupport
import SummaryWriter
import PythonKit

let metrics = Python.import("sklearn.metrics")

// let runName = "resnet_run_3" // resnet
let runName = "transf_run1" // transformer
// let batchSize = 200 // resnet
let batchSize = 100 // transformer
let maxSequenceLength =  500
let nEpochs = 20
// let learningRate: Float = 1e-3 // resnet
let learningRate: Float = 2e-5 // transformer
let logdir = "tboard/Motion2label/\(runName)"
let balanceClassSamples: Int? = 5900
let minMotionLength = 20 // 2 secs. (for downsampled motion)

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")
print("logdir: \(logdir)")
print("balanceClassSamples: \(String(describing:balanceClassSamples))")
print("minMotionLength: \(minMotionLength)")

let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.norm.10Hz.plist")
let labelsURL = dataURL.appendingPathComponent("labels_ds_v2.csv")

// X10 warmup
let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
let eagerTensorSum = eagerTensor1 + eagerTensor2
print(eagerTensorSum)
print(eagerTensor1.device)
let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
print(x10Tensor2.device)

print("\nLoading dataset...")
print(serializedDatasetURL.path)
print(labelsURL.path)
let dataset = try! Motion2Label(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize,
    balanceClassSamples: balanceClassSamples,
    minMotionLength: minMotionLength
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

// instantiate DenseMotionClassifier
var hiddenLayerCount: Int = 4 //12
var attentionHeadCount: Int = 4 //12
var hiddenSize = 96*attentionHeadCount // 64*12 = 768
var intermediateSize: Int = hiddenSize*4 // 3072/768=4
let classCount = 5

func getDenseMotionClassifier(
        hiddenLayerCount: Int, 
        attentionHeadCount: Int, 
        hiddenSize: Int, 
        intermediateSize: Int, 
        classCount: Int
    ) -> DenseMotionClassifier {

    let caseSensitive: Bool = false
    let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")

    let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
    let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary,
        caseSensitive: caseSensitive, unknownToken: "[UNK]", maxTokenLength: nil)

    let variant: BERT.Variant = .bert          

    let transformerEncoder = FeatureTransformerEncoder(
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

    let inputSize = dataset.motionDataset.motionSamples[0].motionFramesArray.shape[1]
    return DenseMotionClassifier(transformerEncoder: transformerEncoder, inputSize: inputSize, classCount: classCount, maxSequenceLength: maxSequenceLength)
}

// instantiate DenseMotionClassifier
var motionClassifier = getDenseMotionClassifier(
    hiddenLayerCount: hiddenLayerCount, 
    attentionHeadCount: attentionHeadCount, 
    hiddenSize: hiddenSize, 
    intermediateSize: intermediateSize, 
    classCount: classCount
)

// instantiate ResNetMotionClassifier
// var resNetClassifier = ResNet(classCount: hiddenSize, depth: .resNet18, downsamplingInFirstStage: false, channelCount: 1)
// var motionClassifier = ResNetMotionClassifier(resNetClassifier: resNetClassifier, maxSequenceLength: maxSequenceLength)

let device = Device.defaultXLA
print(device)
motionClassifier.move(to: device)

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

// var optimizer = SGD(for: motionClassifier, learningRate: learningRate)
var optimizer = Adam(for: motionClassifier, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

print("optimizer = \(optimizer)")

let logdirURL = dataURL.appendingPathComponent(logdir, isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

print("\nTraining (Dense/ResNet)MotionClassifier for the motion2Label task!")
var trainingStepCount = 0
time() {
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(nEpochs).enumerated() {
        time() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        for batch in epochBatches {
            let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Int32>(batch.label))
            let documents = MotionBatch(copying: eagerDocuments, to: device)
            let labels = Tensor(copying: eagerLabels, to: device)
            let (loss, gradients) = valueWithGradient(at: motionClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingBatchCount += 1
            optimizer.update(&motionClassifier, along: gradients)
            LazyTensorBarrier()
            trainingLossSum += loss.scalarized()
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
        time {
        Context.local.learningPhase = .inference
        var devLossSum: Float = 0
        var devBatchCount = 0
        var correctGuessCount = 0
        var totalGuessCount = 0

        var predictions: [Prediction] = []
        for batch in dataset.validationBatches {
            let valBatchSize = batch.data.motionFrames.shape[0]
            let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Int32>(batch.label))
            let documents = MotionBatch(copying: eagerDocuments, to: device)
            let labels = Tensor(copying: eagerLabels, to: device)

            let logits = motionClassifier(documents)
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
            let probs = softmax(logits, alongAxis: 1)
            let preds = logits.argmax(squeezingAxis: 1)
            let correctPredictions = preds .== labels

            LazyTensorBarrier()

            devLossSum += loss.scalarized()
            devBatchCount += 1
            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += valBatchSize

            // copy tensors to CPU
            let eagerPreds = Tensor(copying: preds, to: Device.defaultTFEager)
            let eagerProbs = Tensor(copying: probs, to: Device.defaultTFEager)

            let batchPreds = (0..<eagerPreds.shape[0]).map { 
                (idx) -> Prediction in
                let classIdx: Int = Int(eagerPreds[idx].scalar!)
                let prob = eagerProbs[idx, classIdx].scalar!
                return Prediction(classIdx: classIdx, className: dataset.labels[classIdx], probability: prob)
            }

            predictions.append(contentsOf: batchPreds)
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

        let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
        let y_pred = predictions.map { $0.className }
        print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
        }
        }
    }
    summaryWriter.flush()
}

print("\nFinal stats:")
let preds = motionClassifier.predict(motionSamples: dataset.testMotionSamples, labels: dataset.labels, batchSize: batchSize, device: device)
let y_true = dataset.testMotionSamples.map { dataset.getLabel($0.sampleID)!.label }
let y_pred = preds.map { $0.className }
print("accuracy: \(metrics.accuracy_score(y_true, y_pred))")
print(dataset.labels)
print(metrics.confusion_matrix(y_pred, y_true, labels: dataset.labels))
print(metrics.classification_report(y_true, y_pred, labels: dataset.labels, zero_division: false))

print("\nFinito.")
