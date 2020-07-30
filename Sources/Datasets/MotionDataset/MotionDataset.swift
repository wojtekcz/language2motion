import Foundation

public class MotionDataset: Codable {
    public let datasetFolderURL: URL
    public var motionSamples: [MotionSample]
    public var maxSampleID = 3966

    public init(datasetFolderURL: URL, grouppedJoints: Bool = true, normalized: Bool = true, sampled: Int? = nil, factor: Int = 1, maxFrames: Int = 50000) {
        self.datasetFolderURL = datasetFolderURL
        var motionSamples: [MotionSample] = []
        let fm = FileManager()
        
        var sampleIDs: [Int] = Array<Int>((0...maxSampleID))
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
                if factor == 1 {
                    let motionSample = MotionSample(sampleID: sampleID, mmmURL: mmmURL, annotationsURL: annotationsURL, grouppedJoints: grouppedJoints, normalized: normalized, maxFrames: maxFrames)
                    motionSamples.append(motionSample)
                } else {
                    let _motionSamples = MotionSample.downsampledMutlipliedMotionSamples(
                        sampleID: sampleID, 
                        mmmURL: mmmURL, 
                        annotationsURL: annotationsURL, 
                        grouppedJoints: grouppedJoints, 
                        normalized: normalized, 
                        factor: factor, 
                        maxFrames: maxFrames
                    )
                    motionSamples.append(contentsOf: _motionSamples)
                }
            } else {
                print("** Sample \(sampleID) doesn't exist.")
            }
        }
        self.motionSamples = motionSamples
        print("motionSamples.count: \(motionSamples.count)")
    }

    public init(datasetFolderURL: URL, motionSamples: [MotionSample]) {
        self.datasetFolderURL = datasetFolderURL
        self.motionSamples = motionSamples
    }

    // TODO: code throwing errors
    public init(from serializedDatasetURL: URL) {
        let data: Data = FileManager.default.contents(atPath: serializedDatasetURL.path)!
        let motionDataset = try! PropertyListDecoder().decode(MotionDataset.self, from: data)
        datasetFolderURL = motionDataset.datasetFolderURL
        motionSamples = motionDataset.motionSamples
        maxSampleID = motionDataset.maxSampleID
    }

    // TODO: code throwing errors
    public func write(to serializedDatasetURL: URL) {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let mdData = try! encoder.encode(self)
        try! mdData.write(to: serializedDatasetURL) 
    }

    public var description: String {
        return "MotionDataset(motionSamples: \(motionSamples.count))"
    }
}

extension MotionDataset {
    public convenience init(datasetFolderURL: URL, sampled: Int? = nil, freq: Int? = 10, maxFrames: Int = 500, maxSampleID: Int = 3966) {
        var motionSamples: [MotionSample] = []
        let fm = FileManager()
        
        var sampleIDs: [Int] = Array<Int>((0...maxSampleID))
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
                    let motionSample = MotionSample(sampleID: sampleID, mmmURL: mmmURL, annotationsURL: annotationsURL, grouppedJoints: false, normalized: false, maxFrames: maxFrames)
                    motionSamples.append(motionSample)
                } else {
                    let _motionSamples = MotionSample.downsampledMutlipliedMotionSamples2(
                        sampleID: sampleID, 
                        mmmURL: mmmURL, 
                        annotationsURL: annotationsURL, 
                        freq: freq!, 
                        maxFrames: maxFrames
                    )
                    motionSamples.append(contentsOf: _motionSamples)
                }
            } else {
                print("** Sample \(sampleID) doesn't exist.")
            }
        }
        print("motionSamples.count: \(motionSamples.count)")
        self.init(datasetFolderURL: datasetFolderURL, motionSamples: motionSamples)
    }
}
