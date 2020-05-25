import Foundation
import Datasets

print("Running MotionData preprocessing...")

let maxSamples: Int = 500
let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.normalized.\(maxSamples).plist")

var date = Date()
let motionData = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: maxSamples, grouppedJoints: false, normalized: true)
print(abs(date.timeIntervalSinceNow))

date = Date()
print("Encoding to property list..., writing to file...")
motionData.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
