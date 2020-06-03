import Foundation
import TensorFlow
import Datasets


let batchSize = 2
let tensorWidth = 60

print("batchSize: \(batchSize)")
print("tensorWidth: \(tensorWidth)")

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.normalized.100.plist")
let labelsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

print("\nLoading dataset...")
let dataset = try! Motion2Label2(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: tensorWidth,
    batchSize: batchSize
) { 
    // TODO: move this to the dataset
    (example: Motion2LabelExample) -> LabeledMotionBatch in
    let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
    let motionFlag = Tensor<Int32>(motionFrames[0..., 44...44])
    let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag)
    let label = Tensor<Int32>(Int32(example.label!.idx))
    return LabeledMotionBatch(
        data: motionBatch, 
        label: label
    )
}

print("dataset.trainingExamples.count: \(dataset.trainingExamples.count)")
print("dataset.validationExamples.count: \(dataset.validationExamples.count)")

// print("dataset.trainingExamples[0]: \(dataset.trainingExamples[0])")

// for (epoch, epochBatches) in dataset.trainingEpochs.prefix(5).enumerated() {
//     print("[Epoch \(epoch + 1)]")
//     for _ in epochBatches {
//         // print(7)
//     }
// }

// print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
// for _ in dataset.validationBatches {
//     // print(8)
// }

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

print("\nFinito.")
