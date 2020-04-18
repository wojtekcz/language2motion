import Foundation

public class MotionDataset {
    let datasetFolderURL: URL
    var motionSamples: [MotionSample] = []

    public init(datasetFolderURL: URL, maxSamples: Int) {
        self.datasetFolderURL = datasetFolderURL
        
        for i in 1...maxSamples {
            let mmmFilename = String(format: "%05d_mmm.xml", i)
            let annotationsFilename = String(format: "%05d_annotations.json", i)
            print("\(i), \(mmmFilename), \(annotationsFilename)")
            
            let mmmURL = datasetFolderURL.appendingPathComponent(mmmFilename)
            let annotationsURL = datasetFolderURL.appendingPathComponent(annotationsFilename)
            
            let motionSample = MotionSample(mmmURL: mmmURL, annotationsURL: annotationsURL)            
            motionSamples.append(motionSample)
        }
    }
    
    func describe() -> String {
        return "MotionDataset(motionSamples: \(motionSamples.count))"
    }
}
