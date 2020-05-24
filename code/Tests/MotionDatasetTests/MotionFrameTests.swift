import XCTest
import class Foundation.Bundle
import Datasets


final class MotionDatasetTests: XCTestCase {
    let maxSamples: Int = 4000
    var serializedDatasetURL: URL {
        return URL(fileURLWithPath: "/notebooks/language2motion.gt/data/motion_dataset.\(maxSamples).plist")
    }

    public var description: String {
        return "MotionDataset(motionSamples: \(maxSamples))"
    }

    func serializeMotionSample() throws {
        let mmmURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/00186_mmm.xml")
        let annotationsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/00186_annotations.json")
        let m186 = MotionSample(sampleID: 186, mmmURL: mmmURL, annotationsURL: annotationsURL)
        print(m186.description)
        print(m186.annotations)
        // serialize sample to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        let msJSON = try encoder.encode(m186)
        print(String(data: msJSON, encoding: .utf8)!)

        // decode from JSON
        let decoder = JSONDecoder()
        let ms2 = try decoder.decode(MotionSample.self, from: msJSON)
        print(ms2)
    }

    func serializeMotionDataJSON() throws {
        let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")

        var date = Date()
        let motionData = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: 10)
        print(abs(date.timeIntervalSinceNow))

        // serialize dataset to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        let mdJSON = try encoder.encode(motionData)
        let jsonStr = String(data: mdJSON, encoding: .utf8)!
        // print(jsonStr)
        print(jsonStr.count)
        let fileURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/dataset.json")
        try jsonStr.write(to: fileURL, atomically: true, encoding: String.Encoding.utf8)
        // try JSONSerialization.data(withJSONObject: mdJSON)
        // .write(to: fileURL)

        // decode from JSON
        print("Decoding...")
        let decoder = JSONDecoder()
        date = Date()
        let md2 = try decoder.decode(MotionData.self, from: mdJSON)
        print(md2.description)
        print(abs(date.timeIntervalSinceNow))
    }

    func serializeMotionDataBinary() throws {
        let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")

        var date = Date()
        let motionData = MotionData(datasetFolderURL: datasetFolderURL, maxSamples: maxSamples)
        print(abs(date.timeIntervalSinceNow))

        date = Date()
        print("Encoding to property list..., writing to file...")
        motionData.write(to: serializedDatasetURL)
        print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
    }

    func readBinaryMotionData() throws {
        print("Reading..., decoding...")
        let date = Date() 
        let motionData = MotionData(from: serializedDatasetURL)
        print("Done in \(abs(date.timeIntervalSinceNow)) sec.")
        print(motionData.description)
    }

    static var allTests = [
        ("serializeMotionSample", serializeMotionSample),
        ("serializeMotionDataJSON", serializeMotionDataJSON),
        ("serializeMotionDataBinary", serializeMotionDataBinary),
        ("readBinaryMotionData", readBinaryMotionData),
    ]
}
