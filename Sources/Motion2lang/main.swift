import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionLangModels

/// Set training params
let runName = "run_2"
//let batchSize = 10
let batchSize = 100
let maxMotionLength = 50
let maxTextSequenceLength = 20
let nEpochs = 150
let learningRate: Float = 5e-4

print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxMotionLength: \(maxMotionLength)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("nEpochs: \(nEpochs)")
print("learningRate: \(learningRate)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.plist")

/// Select eager or X10 backend

// let device = Device.defaultXLA
let device = Device.defaultTFEager
print("backend: \(device)")


/// X10 warm-up
let eagerTensor1 = Tensor([0.0, 1.0, 2.0])
let eagerTensor2 = Tensor([1.5, 2.5, 3.5])
let eagerTensorSum = eagerTensor1 + eagerTensor2
//print(eagerTensorSum)
//print(eagerTensor1.device)
let x10Tensor2 = Tensor([1.5, 2.5, 3.5], on: Device.defaultXLA)
//print(x10Tensor2.device)

// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = LegacyTextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)

// instantiate model
let inputSize = 47 // TODO: get value from dataset
let targetVocabSize = vocabulary.count
let layerCount: Int = 6
let modelSize: Int = 128
let feedForwardSize: Int = 512
let headCount: Int = 4
let dropoutProbability: Double = 0.1

var model = MotionLangTransformer(
    inputSize: inputSize,
    targetVocabSize: targetVocabSize,
    layerCount: layerCount, 
    modelSize: modelSize, 
    feedForwardSize: feedForwardSize, 
    headCount: headCount, 
    dropoutProbability: dropoutProbability
)

model.move(to: device)

/// load dataset
print("\nLoading dataset...")

var dataset = try Motion2Lang(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 20,
    maxMotionLength: 100,
    trainTestSplit: 0.9
) { (motionSample: MotionSample) -> MotionLangBatch in
    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    return singleBatch
}

print("Dataset acquired.")

/// Test model with one batch
// get a batch
//print("\nOne batch (MotionLangBatch):")
//var epochIterator = dataset.trainingEpochs.enumerated().makeIterator()
//let epoch = epochIterator.next()
//let batches = Array(epoch!.1)
//let batch: MotionLangBatch = batches[0]
//print("type: \(type(of:batch))")
//print("motionFrames.shape: \(batch.motionFrames.shape)")
////print("motionFlag.shape: \(batch.motionFlag.shape)")
//print("mask.shape: \(batch.mask.shape)")
//print("origMotionFramesCount.shape: \(batch.origMotionFramesCount.shape)")
//print("origMotionFramesCount: \(batch.origMotionFramesCount)")
//print("targetTokenIds.shape: \(batch.targetTokenIds.shape)")
//print("targetMask.shape: \(batch.targetMask.shape)")
//print("targetTruth.shape: \(batch.targetTruth.shape)")

// run one batch
//print("\nRun one batch:")
//print("==============")
//let deviceBatch = MotionLangBatch(copying: batch, to: device)
//let output = model(deviceBatch)
//print("output.shape: \(output.shape)")

/// Optimizer
var optimizer = Adam(for: model, learningRate: learningRate)
optimizer = Adam(copying: optimizer, to: device)

let logdirURL = dataURL.appendingPathComponent("runs/Motion2lang/\(runName)", isDirectory: true)
let summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)

/// Training helpers
func update(model: inout MotionLangTransformer, using optimizer: inout Adam<MotionLangTransformer>, for batch: MotionLangBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let result = withLearningPhase(.training) { () -> Float in
        let (loss, grad) = valueWithGradient(at: model) {
            (model) -> Tensor<Float> in
            let logits = model.generate(input: batch).reshaped(to: [resultSize, -1])
            let sce = softmaxCrossEntropy(logits: logits, labels: labels)
            return sce
        }
        optimizer.update(&model, along: grad)
        LazyTensorBarrier()
        return loss.scalarized()
    }
    return result
}

/// returns validation loss
func validate(model: inout MotionLangTransformer, for batch: MotionLangBatch) -> Float {
    let labels = batch.targetTruth.reshaped(to: [-1])
    let resultSize = batch.targetTruth.shape.last! * batch.targetTruth.shape.first!
    let result = withLearningPhase(.inference) { () -> Float in
        softmaxCrossEntropy(logits: model.generate(input: batch).reshaped(to: [resultSize, -1]), labels: labels).scalarized()
    }
    LazyTensorBarrier()
    return result
}

/// Set up decoding
func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input)
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    for _ in 0..<maxLength {
        let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
        var motionPartMask = MotionLangBatch.makeStandardMask(target: motionPartFlag, pad: 0, shiftRight: true)
        let motionLen = Int(motionPartFlag.sum().scalar!)
        motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
        motionPartMask = abs(motionPartMask)
        
        let decoderInput = MotionLangBatch(motion: input.motion,
                                     mask: input.mask,
                                     origMotionFramesCount: input.origMotionFramesCount,
                                     targetTokenIds: ys,
                                     targetMask: motionPartMask,
                                     targetTruth: input.targetTruth)
        let out = model.decode(input: decoderInput, memory: memory)
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1) // , on: device
    }
    return ys
}

func greedyDecodeSample(_ sample_id: Int, maxLength: Int = 15) {
    let motionSample = dataset.motionSampleDict[sample_id]!
    print("\nsample: \(motionSample.sampleID), \"\(motionSample.annotations[0])\", motion: \(motionSample.timesteps[-1]) sec (\(motionSample.motion.shape[0]) frames)")

    let singleExampleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    var source = MotionLangBatch.reduceDataBatches([singleExampleBatch])
    source = MotionLangBatch(copying: source, to: Device.defaultTFEager)
    let out = greedyDecode(model: model, input: source, maxLength: maxLength, startSymbol: textProcessor.bosId)
    let outputStr = textProcessor.decode(tensor: out)
    print("decoded: \"\(outputStr)\"")
}

let samplesToDecode = [
    ["sampleID": 449, "text": "A person runs forward."],
    ["sampleID": 3921, "text": "A human is swimming."],
    ["sampleID": 843, "text": "A person walks."],
    ["sampleID": 1426, "text": "A person plays the air guitar."],
    ["sampleID": 1292, "text": "A person performs a squat."],
    ["sampleID": 1315, "text": "A human raises their left foot and touches it with the right hand."]
]

/// Training loop
print("\nTraining Transformer for the Motion2lang task!")
var trainingStepCount = 0
time() {
    LazyTensorBarrier()
    for (epoch, epochBatches) in dataset.trainEpochs.prefix(nEpochs).enumerated() {
        print("\n[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        if epoch == 0 {
            print("epochBatches.count: \(epochBatches.count)")
        }

        for eagerBatch in epochBatches {
            if (trainingStepCount < 5) {
                print("==> step \(trainingStepCount)")
            }
            let batch = MotionLangBatch(copying: eagerBatch, to: device)
            let loss: Float = update(model: &model, using: &optimizer, for: batch)
            if (trainingStepCount < 5) {
                print("current loss at step \(trainingStepCount): \(loss)")
            }
            trainingLossSum += loss
            trainingBatchCount += 1
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
            print("dataset.testBatches.count: \(dataset.testBatches.count)")
        }
        Context.local.learningPhase = .inference
        var devLossSum: Float = 0
        var devBatchCount = 0
        var totalGuessCount = 0

        for eagerBatch in dataset.testBatches {
            let batch = MotionLangBatch(copying: eagerBatch, to: device)
            let loss: Float = validate(model: &model, for: batch)
            let valBatchSize = batch.motion.shape[0]

            devLossSum += loss
            devBatchCount += 1
            totalGuessCount += valBatchSize
        }

        print(
            """
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
        summaryWriter.writeScalarSummary(tag: "EpochTestLoss", step: epoch+1, value: devLossSum / Float(devBatchCount))

        Context.local.learningPhase = .inference
        model.move(to: Device.defaultTFEager)
        for sample in samplesToDecode {
            greedyDecodeSample(sample["sampleID"] as! Int, maxLength: 15)
        }
        model.move(to: device)
    }
    summaryWriter.flush()
}

print("\nFinished training.")

/// Generate motion description
let sample_id = 446
greedyDecodeSample(sample_id)

print("\nFinito.")
