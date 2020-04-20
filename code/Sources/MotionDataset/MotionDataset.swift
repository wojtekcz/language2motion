import Foundation

public class MotionDataset {
    let datasetFolderURL: URL
    var motionSamples: [MotionSample] = []

    public init(datasetFolderURL: URL, maxSamples: Int) {
        self.datasetFolderURL = datasetFolderURL
        let fm = FileManager()
        
        for i in 1...maxSamples {
            let mmmFilename = String(format: "%05d_mmm.xml", i)
            let annotationsFilename = String(format: "%05d_annotations.json", i)
            print("Sample \(i), \(mmmFilename), \(annotationsFilename)")
            
            let mmmURL = datasetFolderURL.appendingPathComponent(mmmFilename)
            let annotationsURL = datasetFolderURL.appendingPathComponent(annotationsFilename)
            
            if fm.fileExists(atPath: mmmURL.path) {
                let motionSample = MotionSample(sampleID: i, mmmURL: mmmURL, annotationsURL: annotationsURL)            
                motionSamples.append(motionSample)
            } else {
                print("** Sample \(i) doesn't exist.")
            }
        }
    }
    
    func describe() -> String {
        return "MotionDataset(motionSamples: \(motionSamples.count))"
    }
}
