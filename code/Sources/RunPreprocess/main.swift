import Foundation
import MotionDataset

print("Running MotionDataset preprocessing...")

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/m2l.gt/data/2017-06-22/")

let date = Date()
var motionDataset = MotionDataset(datasetFolderURL: datasetFolderURL, maxSamples: 100)
print(abs(date.timeIntervalSinceNow))
