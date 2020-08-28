import Foundation
import Datasets

let sampled: Int? = 100 // nil
let freq: Int? = 10
let maxFrames = 500

let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")
let sampledStr = (sampled != nil) ? "sampled." : ""
let freqStr = (freq != nil) ? "\(freq!)Hz." : ""

print("Running MotionData preprocessing (\(String(describing:sampled)))...")

var date = Date()
let motionDataset = MotionDataset2Processor.loadDatasetFromFolder(datasetFolderURL: datasetFolderURL, sampled: sampled, freq: freq, maxFrames: maxFrames)
print(abs(date.timeIntervalSinceNow))

let numberStr = "\(motionDataset.motionSamples.count)."
let serializedDatasetURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset_v3.\(freqStr)\(sampledStr)\(numberStr)plist")

date = Date()
print("Encoding to property list..., writing to file '\(serializedDatasetURL.path)'")
motionDataset.write(to: serializedDatasetURL)
print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
