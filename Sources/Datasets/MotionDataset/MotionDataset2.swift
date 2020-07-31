import Foundation

public enum DatasetSize: String {
    case full = ""
    case midi = "midi."
    case mini = "mini."
}

public class MotionDataset2: Codable {
    public let datasetFolderURL: URL
    public var motionSamples: [MotionSample2]

    public init(datasetFolderURL: URL, motionSamples: [MotionSample2]) {
        self.datasetFolderURL = datasetFolderURL
        self.motionSamples = motionSamples
    }

    public init(from serializedDatasetURL: URL) {
        let data: Data = FileManager.default.contents(atPath: serializedDatasetURL.path)!
        let motionDataset = try! PropertyListDecoder().decode(MotionDataset2.self, from: data)
        datasetFolderURL = motionDataset.datasetFolderURL
        motionSamples = motionDataset.motionSamples
    }

    public func write(to serializedDatasetURL: URL) {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let mdData = try! encoder.encode(self)
        try! mdData.write(to: serializedDatasetURL) 
    }

    public var description: String {
        return "MotionDataset2(motionSamples: \(motionSamples.count))"
    }
}
