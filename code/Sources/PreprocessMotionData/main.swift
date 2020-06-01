import Foundation
import Datasets

let maxSamples: Int = 500
let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.grouppedJoints.normalized.\(maxSamples).plist")

print("Running MotionData preprocessing (maxSamples=\(maxSamples))...")

var date = Date()
let motionData = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: maxSamples, grouppedJoints: true, normalized: true)
print(abs(date.timeIntervalSinceNow))

date = Date()
print("Encoding to property list..., writing to file \(serializedDatasetURL)")
motionData.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
