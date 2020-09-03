import Foundation


public struct MotionDataset2Processor {
    public static let maxSampleID = 3966

    public static func loadDatasetFromFolder(datasetFolderURL: URL, sampled: Int? = nil, freq: Int? = 10, maxFrames: Int = 500, multiply: Bool = true) -> MotionDataset {
        var motionSamples: [MotionSample] = []
        let fm = FileManager()
        
        var sampleIDs: [Int] = Array<Int>((0...MotionDataset2Processor.maxSampleID))
        if sampled != nil {
            sampleIDs = Array(sampleIDs.choose(sampled!))
        }
        
        for sampleID in sampleIDs {
            let mmmFilename = String(format: "%05d_mmm.xml", sampleID)
            let annotationsFilename = String(format: "%05d_annotations.json", sampleID)
            print("Sample \(sampleID), \(mmmFilename), \(annotationsFilename)")
            
            let mmmURL = datasetFolderURL.appendingPathComponent(mmmFilename)
            let annotationsURL = datasetFolderURL.appendingPathComponent(annotationsFilename)
            
            if fm.fileExists(atPath: mmmURL.path) {
                if freq == nil {
                    let motionSample = LegacyMMMReader.motionSampleFromMMM(sampleID: sampleID, mmmURL: mmmURL, annotationsURL: annotationsURL, maxFrames: maxFrames)
                    motionSamples.append(motionSample)
                } else {
                    var _motionSamples = LegacyMMMReader.downsampledMutlipliedMotionSamples(
                        sampleID: sampleID, 
                        mmmURL: mmmURL, 
                        annotationsURL: annotationsURL, 
                        freq: freq!, 
                        maxFrames: maxFrames
                    )
                    if !multiply {
                        _motionSamples = Array(_motionSamples[0..<1])
                    }
                    motionSamples.append(contentsOf: _motionSamples)
                }
            } else {
                print("** Sample \(sampleID) doesn't exist.")
            }
        }
        print("motionSamples.count: \(motionSamples.count)")        
        return MotionDataset(datasetFolderURL: datasetFolderURL, motionSamples: motionSamples)
    }
}
