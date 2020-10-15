import Foundation

public enum DatasetSize: String {
    case full = ""
    case midi = "midi."
    case mini = "mini."
    case micro = "micro."

    case multi_full = "multi."
    case multi_midi = "multi.midi."
    case multi_mini = "multi.mini."
    case multi_micro = "multi.micro."

    case same_midi = "same.midi."
    case same_mini = "same.mini."
    case same_micro = "same.micro."

    //   2 - small_micro
    //   2 - small_multi_micro
    //  10 - small_multi_mini
    // 100 - small_multi_midi
    case small_micro = "small.micro."
    case small_multi_micro = "small.multi.micro."
    case small_multi_mini = "small.multi.mini."
    case small_multi_midi = "small.multi.midi."
}

public class MotionDataset: Codable {
    public let datasetFolderURL: URL
    public var motionSamples: [MotionSample]

    public init(datasetFolderURL: URL, motionSamples: [MotionSample]) {
        self.datasetFolderURL = datasetFolderURL
        self.motionSamples = motionSamples
    }

    public init(from serializedDatasetURL: URL) {
        let data: Data = FileManager.default.contents(atPath: serializedDatasetURL.path)!
        let motionDataset = try! PropertyListDecoder().decode(Self.self, from: data)
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
        return "MotionDataset(motionSamples: \(motionSamples.count))"
    }
}
