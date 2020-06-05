import Foundation
import TensorFlow
import Datasets
import MotionModels
import ImageClassificationModels
import TextModels
import ModelSupport


let batchSize = 25
let maxSequenceLength = 600

print("batchSize: \(batchSize)")
print("maxSequenceLength: \(maxSequenceLength)")

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset_v2.normalized.plist")
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
var hiddenSize = 768
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
var hiddenLayerCount: Int = 12
var attentionHeadCount: Int = 12
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
print("\nOne batch (LabeledMotionBatch):")
var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
let epoch = epochIterator.next()
let batches = Array(epoch!.1)
let batch = batches[0]
print("type: \(type(of:batch))")
print("\nOne motionBatch")
let motionBatch = batch.data
print("type: \(type(of:motionBatch))")
print("motionFrames.shape: \(motionBatch.motionFrames.shape)")
print("motionFlag.shape: \(motionBatch.motionFlag.shape)")


// run one batch
print("\nRun one batch:")
print("==============")
let classifierOutput = motionClassifier(motionBatch)
print("classifierOutput.shape: \(classifierOutput.shape)")

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

let optimizer = SGD(for: motionClassifier, learningRate: 1e-5)

print("\nTraining MotionClassifier for the motion2Label task!")
time() {
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(5).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        print("epochBatches.count: \(epochBatches.count)")

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
            optimizer.update(&motionClassifier, along: gradients)
            // LazyTensorBarrier()

        }
        print(
            """
            Training loss: \(trainingLossSum / Float(trainingBatchCount))
            """
        )

        print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
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
        
        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
    }
}

print("\nFinito.")
