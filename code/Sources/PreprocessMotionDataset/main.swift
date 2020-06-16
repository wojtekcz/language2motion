import Foundation
import Datasets

let grouppedJoints = false
let normalized = true
let sampled: Int? = 500 // 500

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let grouppedJointsStr = grouppedJoints ? "grouppedJoints." : ""
let normalizedStr = normalized ? "normalized." : ""
var sampledStr = "" 
if sampled != nil { 
    sampledStr = "sampled.\(sampled!)." 
}

let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.motion_flag.\(grouppedJointsStr)\(normalizedStr)\(sampledStr)plist")

print("Running MotionData preprocessing (sampledStr=\(sampledStr))...")

var date = Date()
let motionDataset = MotionDataset(datasetFolderURL: datasetFolderURL, grouppedJoints: grouppedJoints, normalized: true, sampled: sampled)
print(abs(date.timeIntervalSinceNow))

date = Date()
print("Encoding to property list..., writing to file '\(serializedDatasetURL.path)'")
motionDataset.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
