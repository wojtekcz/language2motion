import Foundation
import Datasets

let maxSamples: Int = 100
let grouppedJoints = false
let normalized = true

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let grouppedJointsStr = grouppedJoints ? "grouppedJoints." : ""
let normalizedStr = normalized ? "normalized." : ""
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.\(grouppedJointsStr)\(normalizedStr)\(maxSamples).plist")

print("Running MotionData preprocessing (maxSamples=\(maxSamples))...")

var date = Date()
let motionData = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: maxSamples, grouppedJoints: grouppedJoints, normalized: true)
print(abs(date.timeIntervalSinceNow))

date = Date()
print("Encoding to property list..., writing to file '\(serializedDatasetURL.path)'")
motionData.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
