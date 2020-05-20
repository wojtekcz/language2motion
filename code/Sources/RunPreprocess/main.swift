import Foundation
import MotionDataset

print("Running MotionData preprocessing...")

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/m2l.gt/data/2017-06-22/")

let date = Date()
var motionDatas = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: 100)
print(abs(date.timeIntervalSinceNow))
