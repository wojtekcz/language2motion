import Foundation
import Datasets

let grouppedJoints = false
let normalized = true
let sampled: Int? = nil // nil
// let factor = 1
// let maxFrames = 50000

let factor = 10
let maxFrames = 5000

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let grouppedJointsStr = grouppedJoints ? "grouppedJoints." : ""
let normalizedStr = normalized ? "normalized." : ""
let sampledStr = (sampled != nil) ? "sampled." : ""
let downsampledStr = factor>1 ? "downsampled." : ""

// let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.\(grouppedJointsStr)\(normalizedStr)\(downsampledStr)\(sampledStr)plist")

print("Running MotionData preprocessing (\(String(describing:sampled)))...")

var date = Date()
let motionDataset = MotionDataset(datasetFolderURL: datasetFolderURL, grouppedJoints: grouppedJoints, normalized: true, sampled: sampled, factor: factor, maxFrames: maxFrames)
print(abs(date.timeIntervalSinceNow))

let numberStr = "\(motionDataset.motionSamples.count)."
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.\(grouppedJointsStr)\(normalizedStr)\(downsampledStr)\(sampledStr)\(numberStr)plist")

date = Date()
print("Encoding to property list..., writing to file '\(serializedDatasetURL.path)'")
motionDataset.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
