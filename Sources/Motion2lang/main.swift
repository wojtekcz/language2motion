import TensorFlow
import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import MotionLangModels
import TrainingLoop
import x10_optimizers_optimizer

/// Set training params
let batchSize = 10
let runName = "run_9"
//let batchSize = 300
let maxMotionLength = 50
let maxTextSequenceLength = 40
let nEpochs = 150
let peakLearningRate: Float = 5e-4

let stepsPerEpoch = 1967/batchSize*2 // function of training set size and batching configuration

let beta1: Float = 0.9
let beta2: Float = 0.999
let useBiasCorrection = false

//let datasetSize: DatasetSize = .multi_full
let datasetSize: DatasetSize = .micro


print("runName: \(runName)")
print("batchSize: \(batchSize)")
print("maxMotionLength: \(maxMotionLength)")
print("maxTextSequenceLength: \(maxTextSequenceLength)")
print("nEpochs: \(nEpochs)")
print("peakLearningRate: \(peakLearningRate)")
print("datasetSize: \(datasetSize)")
print("stepsPerEpoch: \(stepsPerEpoch)")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif
let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

let logdirURL = dataURL.appendingPathComponent("runs/Motion2lang/", isDirectory: true)
let rundirURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
let checkpointURL = rundirURL.appendingPathComponent("checkpoints", isDirectory: true)

// FIXME: how to make macOS builds use filesystem in read/write mode?
#if os(Linux)
    try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
#endif

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

// The following is a workaround needed until X10 can set log levels and memory growth parameters.
// let _ = _ExecutionContext.global

/// load dataset
print("\nLoading dataset...")

var dataset = try Motion2Lang(
    motionDatasetURL: motionDatasetURL,
    batchSize: batchSize,
    minMotionLength: 20,
    maxMotionLength: 100,
    trainTestSplit: 0.9,
    device: device
) { (motionSample: MotionSample) -> MotionLangBatch in
    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    return singleBatch
}

print("Dataset acquired.")

print(dataset.motionSamples.count)

let stepsPerEpoch = dataset.motionSamples.count/batchSize // function of training set size and batching configuration
let lrSlopeMultiplier = 2

/// instantiate text processor
let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = LegacyTextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)

/// instantiate model
print("instantiate model")
let modelSize = 128
let config = MotionLangTransformerConfig(
    vocabSize: vocabulary.count,
    nbJoints: 47,
    layerCount: 6,
    modelSize: modelSize,
    feedForwardSize: 512,
    headCount: 4,
    dropoutProbability: 0.1,
    sentenceMaxPositionalLength: 100,
    motionMaxPositionalLength: 500
)

var start_epoch = 0

/// create new model
var model = MotionLangTransformer(config: config)

/// load model checkpoint
//print("logdirURL: \(logdirURL.path)")
//start_epoch = 10
//let modeName = "model.e\(start_epoch)"
//let modeName = "model.final"
//var model = try! MotionLangTransformer(checkpoint: logdirURL.appendingPathComponent("run_8/checkpoints"), config: config, name: modeName)


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
//var optimizer = Adam(for: model, learningRate: learningRate)

var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
    for: model,
    TensorVisitorPlan(model.differentiableVectorView),
    defaultOptimizer: makeWeightDecayedAdam(
      learningRate: peakLearningRate,
      beta1: beta1,
      beta2: beta2
    )
)

var scheduledLearningRate = LinearlyDecayedParameter(
  baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter<Float>(peakLearningRate),
      warmUpStepCount: 20,
      warmUpOffset: 0),
  slope: -(peakLearningRate / Float(stepsPerEpoch * nEpochs * lrSlopeMultiplier)),  // The LR decays linearly to zero.
  startStep: 10
)

public func learningRateUpdater<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
    if event == .updateStart {
        let optimizer: GeneralOptimizer<MotionLangTransformer> = loop.optimizer as! GeneralOptimizer<MotionLangTransformer>
        let step = optimizer.step + 1 // for scheduled rates and bias correction, steps start at 1
        optimizer.learningRate = scheduledLearningRate(forStep: UInt64(step))
        if useBiasCorrection {
          let f_step = Float(step)
          optimizer.learningRate *= sqrtf(1 - powf(beta2, f_step)) / (1 - powf(beta1, f_step))
        }
        // print("\noptimizer: step: \(optimizer.step), learningRate: \(optimizer.learningRate)")
    }
}

/// stats recorder
public class StatsRecorder {
    let summaryWriter = SummaryWriter(logdir: rundirURL, flushMillis: 30*1000)
    public var trainingStepCount = 0
    public var trainingBatchCount = 0
    public var trainingLossSum: Float = 0.0

    public var validationStepCount = 0
    public var validationBatchCount = 0
    public var validationLossSum: Float = 0.0

    public var epochIndex = 0 // FIXME: Workaround
    public var inValidationPhase = false

    public func writeStats<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
        
        // TODO: write learning rate
        
        if event == .validationStart {
            inValidationPhase = true
            validationBatchCount = 0
            validationLossSum = 0.0
        }

        if event == .validationEnd {
            inValidationPhase = false
            let current_epoch = epochIndex + 1
            let epochValidationLoss = validationLossSum / Float(validationBatchCount)
            summaryWriter.writeScalarSummary(tag: "EpochValidationLoss", step: current_epoch, value: epochValidationLoss)
        }

        if event == .batchEnd {
            guard
            // let batchIndex = loop.batchIndex,
            let lastLoss = loop.lastLoss else {
                return
            }
            if !inValidationPhase {
                // print("\nbatch stats: batchIndex: \(batchIndex), trainingStepCount: \(trainingStepCount), trainingLoss: \(lastLoss)")
                summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: lastLoss.scalar!)
                
                let optimizer: GeneralOptimizer<MotionLangTransformer> = loop.optimizer as! GeneralOptimizer<MotionLangTransformer>
                summaryWriter.writeScalarSummary(tag: "LearningRate", step: trainingStepCount, value: optimizer.learningRate)

                trainingStepCount += 1
                trainingBatchCount += 1
                trainingLossSum += Float(lastLoss.scalar!)
            } else {
                validationStepCount += 1
                validationBatchCount += 1
                validationLossSum += Float(lastLoss.scalar!)
                summaryWriter.writeScalarSummary(tag: "ValidationLoss", step: validationStepCount, value: lastLoss.scalar!)
            }
        }
        if event == .trainingStart {
            trainingBatchCount = 0
            trainingLossSum = 0.0
        }
        if event == .trainingEnd {
            // guard let epochIndex = loop.epochIndex else {
            //     return
            // }
            let current_epoch = epochIndex + 1
            let epochTrainingLoss = trainingLossSum / Float(trainingBatchCount)
            // print("\nepoch stats: current_epoch: \(current_epoch), epochTrainingLoss: \(epochTrainingLoss)")
            summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: current_epoch, value: epochTrainingLoss)
        }
        if event == .fitEnd {
            summaryWriter.flush()
        }
    }
}

let statsRecorder = StatsRecorder()

@differentiable(wrt: y_pred)
func embeddedSoftmaxCrossEntropy(y_pred: Tensor<Float>, y_true: MotionLangBatch.MLTarget) -> Tensor<Float> {
    let resultSize = y_true.targetTruth.shape.last! * y_true.targetTruth.shape.first!
    let logits = y_pred.reshaped(to: [resultSize, -1])
    let labels = y_true.targetTruth.reshaped(to: [-1])
    // TODO: ignore padded entries
    return softmaxCrossEntropy(logits: logits, labels: labels)
}

/// Set up decoding
func greedyDecode(model: MotionLangTransformer, input: MotionLangBatch.MLSource, maxLength: Int, startSymbol: Int32) -> Tensor<Int32> {
    let memory = model.encode(input: input).lastLayerOutput
    var ys = Tensor(repeating: startSymbol, shape: [1,1])
    for _ in 0..<maxLength {
        let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
        var motionPartMask = MotionLangBatch.makeStandardMask(target: motionPartFlag, pad: 0, shiftRight: true)
        let motionLen = Int(motionPartFlag.sum().scalar!)
        motionPartMask[0, 0..<motionLen-1, 0..<motionLen] -= 1
        motionPartMask = abs(motionPartMask)

        let decoderInput = MotionLangBatch.MLSource(sampleID: input.sampleID, motion: input.motion,
                                     mask: input.mask,
                                     origMotionFramesCount: input.origMotionFramesCount,
                                     targetTokenIds: ys,
                                     targetMask: motionPartMask
                                     )
        let out = model.decode(input: decoderInput, memory: memory).lastLayerOutput
        let prob = model.generate(input: out[0...,-1])
        let nextWord = Int32(prob.argmax().scalarized())
        ys = Tensor(concatenating: [ys, Tensor(repeating: nextWord, shape: [1,1])], alongAxis: 1) // , on: device
    }
    return ys
}

func greedyDecodeSample(_ sample_id: Int, maxLength: Int = 15) {
    let motionSample = dataset.motionSampleDict[sample_id]!
    print("\nsample: \(motionSample.sampleID), \"\(motionSample.annotations[0])\", motion: \(motionSample.timesteps[-1]) sec (\(motionSample.motion.shape[0]) frames)")

    let singleBatch = textProcessor.preprocess(motionSample: motionSample, maxMotionLength: maxMotionLength, maxTextSequenceLength: maxTextSequenceLength)
    let out = greedyDecode(model: model, input: singleBatch.source, maxLength: maxLength, startSymbol: textProcessor.bosId)
    let outputStr = textProcessor.decode(tensor: out)
    print("decoded: \"\(outputStr)\"")
}

let samplesToDecode = [
//    ["sampleID": 733, "text": "Ala ma kota."], // for .micro dataset
    ["sampleID": 449, "text": "A person runs forward."],
    ["sampleID": 3921, "text": "A human is swimming."],
    ["sampleID": 843, "text": "A person walks."],
    ["sampleID": 1426, "text": "A person plays the air guitar."],
    ["sampleID": 1292, "text": "A person performs a squat."],
    ["sampleID": 1315, "text": "A human raises their left foot and touches it with the right hand."]
]

// Training loop
print("\nSetting up the training loop")
let trainingProgress = TrainingProgress(metrics: [.loss])
var trainingLoop: TrainingLoop = TrainingLoop(
    training: dataset.trainEpochs,
    validation: dataset.testBatches,
    optimizer: optimizer,
    lossFunction:  embeddedSoftmaxCrossEntropy,
    callbacks: [trainingProgress.update, statsRecorder.writeStats, learningRateUpdater]
)

print("\nTraining Transformer for the Motion2lang task!")
// FIXME: epoch loop workaround for checkpoint saving
for epochIndex in start_epoch..<start_epoch+nEpochs {
    print("epoch \(epochIndex+1)/\(start_epoch + nEpochs)")
    statsRecorder.epochIndex = epochIndex
    try! trainingLoop.fit(&model, epochs: 1, on: device)
    try! model.writeCheckpoint(to: checkpointURL, name: "model.e\(epochIndex+1)")

    Context.local.learningPhase = .inference
    model.move(to: Device.defaultTFEager)
    for sample in samplesToDecode {
        greedyDecodeSample(sample["sampleID"] as! Int, maxLength: 20)
    }
    model.move(to: device)
}

try! model.writeCheckpoint(to: checkpointURL, name: "model.final")
print("\nFinished training.")

/// Generate motion description
let sample_id = 733
greedyDecodeSample(sample_id, maxLength: 20)

print("\nFinito.")
