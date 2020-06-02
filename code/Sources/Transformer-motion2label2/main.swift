import Foundation
import TensorFlow
import Datasets


let batchSize = 2
let tensorWidth = 60

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.normalized.500.plist")
let labelsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

let dataset = try! Motion2Label2(
    serializedDatasetURL: serializedDatasetURL,
    labelsURL: labelsURL,
    maxSequenceLength: tensorWidth,
    batchSize: batchSize
) { 
    (example: Motion2LabelExample) -> LabeledMotionBatch in
    print(1)
    let motionFrames = Tensor<Float>(example.motionSample.motionFramesArray)
    let motionFlag = Tensor<Int32>(motionFrames[0..., 44...44])
    print(motionFlag.shape)
    let motionBatch = MotionBatch(motionFrames: motionFrames, motionFlag: motionFlag)
    let label = Tensor<Int32>(Int32(example.label!.idx))
    return LabeledMotionBatch(
        data: motionBatch, 
        label: label
    )
}

print("dataset.trainingExamples.count: \(dataset.trainingExamples.count)")
print("dataset.validationExamples.count: \(dataset.validationExamples.count)")

print("dataset.trainingExamples[0]: \(dataset.trainingExamples[0])")
print("Finito.")
