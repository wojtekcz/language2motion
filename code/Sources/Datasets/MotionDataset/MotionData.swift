import Foundation

public class MotionData: Codable {
    public let datasetFolderURL: URL
    public let motionSamples: [MotionSample]

    public init(datasetFolderURL: URL, maxSamples: Int, grouppedJoints: Bool = true, normalized: Bool = true) {
        self.datasetFolderURL = datasetFolderURL
        var motionSamples: [MotionSample] = []
        let fm = FileManager()
        
        for i in 1...maxSamples {
            let mmmFilename = String(format: "%05d_mmm.xml", i)
            let annotationsFilename = String(format: "%05d_annotations.json", i)
            print("Sample \(i), \(mmmFilename), \(annotationsFilename)")
            
            let mmmURL = datasetFolderURL.appendingPathComponent(mmmFilename)
            let annotationsURL = datasetFolderURL.appendingPathComponent(annotationsFilename)
            
            if fm.fileExists(atPath: mmmURL.path) {
                let motionSample = MotionSample(sampleID: i, mmmURL: mmmURL, annotationsURL: annotationsURL, grouppedJoints: grouppedJoints, normalized: normalized)            
                motionSamples.append(motionSample)
            } else {
                print("** Sample \(i) doesn't exist.")
            }
        }
        self.motionSamples = motionSamples
    }

    // TODO: code throwing errors
    public init(from serializedDatasetURL: URL) {
        let data: Data = FileManager.default.contents(atPath: serializedDatasetURL.path)!
        let motionData = try! PropertyListDecoder().decode(MotionData.self, from: data)
        datasetFolderURL = motionData.datasetFolderURL
        motionSamples = motionData.motionSamples
    }

    // TODO: code throwing errors
    public func write(to serializedDatasetURL: URL) {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let mdData = try! encoder.encode(self)
        try! mdData.write(to: serializedDatasetURL) 
    }

    public var description: String {
        return "MotionData(motionSamples: \(motionSamples.count))"
    }
}
