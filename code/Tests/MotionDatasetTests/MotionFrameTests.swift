import XCTest
import class Foundation.Bundle
import MotionDataset
import StickyEncoding


extension Data {
    var bytes : [UInt8]{
        return [UInt8](self)
    }
}

final class MotionDatasetTests: XCTestCase {
    // func serializeMotionFrame() throws {
    //     print("serializeMotionFrame")
    //     let mf = MotionFrame(timestamp: 0.0, jointPositions: [0.0, 1.0, 2.0], jointNames: ["ala", "ma", "kota"])
    //     print(mf)
    //     // serialize frame to JSON
    //     let encoder = JSONEncoder()
    //     encoder.outputFormatting = .prettyPrinted

    //     let mfJSON = try encoder.encode(mf)
    //     print(String(data: mfJSON, encoding: .utf8)!)

    //     // decode from JSON
    //     let decoder = JSONDecoder()
    //     let mf2 = try decoder.decode(MotionFrame.self, from: mfJSON)
    //     print(mf2)
    // }

    func serializeMotionSample() throws {
        let mmmURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/00186_mmm.xml")
        let annotationsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/00186_annotations.json")
        let m186 = MotionSample(sampleID: 186, mmmURL: mmmURL, annotationsURL: annotationsURL)
        print(m186.describe())
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

    func serializeMotionDatasetJSON() throws {
        let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")

        var date = Date()
        let motionDataset = MotionDataset(datasetFolderURL: datasetFolderURL, maxSamples: 10)
        print(abs(date.timeIntervalSinceNow))

        // serialize dataset to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        let mdJSON = try encoder.encode(motionDataset)
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
        let md2 = try decoder.decode(MotionDataset.self, from: mdJSON)
        print(md2.describe())
        print(abs(date.timeIntervalSinceNow))
    }

    func serializeMotionDatasetBinary() throws {
        // TODO: save to binary file
        // TODO: read from binary file
        let datasetFolderURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/2017-06-22/")

        var date = Date()
        let motionDataset = MotionDataset(datasetFolderURL: datasetFolderURL, maxSamples: 100)
        print(abs(date.timeIntervalSinceNow))

        // serialize dataset to property list
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        date = Date()
        print("Encoding to property list")
        let mdData = try encoder.encode(motionDataset)
        let fileURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/dataset.plist")
        print("Writing to file")
        try mdData.write(to: fileURL) 
        print(abs(date.timeIntervalSinceNow))

        // // serialize dataset to binary format
        // let encoder = BinaryEncoder()

        // date = Date()
        // print("Encoding to binary format")
        // let bytes = try encoder.encode(motionDataset)
        // let fileURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/dataset.bin2")
        // print("Writing to file")
        // let mdData = Data(bytes)
        // try mdData.write(to: fileURL) 
        // print(abs(date.timeIntervalSinceNow))
    }

    func readBinaryMotionDataset() throws {
        let fileURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/dataset.plist")
        // let fileURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/dataset.bin2")
        // let fileURL = URL(fileURLWithPath: "/root/dataset.bin2")
        print("Reading...")
        let data: Data = FileManager.default.contents(atPath: fileURL.path)!
        // decode from proprty list
        print("Decoding...")
        let date = Date() 
        let motionDataset = try PropertyListDecoder().decode(MotionDataset.self, from: data)
        print("Done...")
        print(motionDataset.describe())
        print(abs(date.timeIntervalSinceNow))

        // decode from binary format
        // print("Decoding...")
        // let decoder = BinaryDecoder()
        // let date = Date() 
        // // let motionDataset = try PropertyListDecoder().decode(MotionDataset.self, from: data)
        // let bytes: [UInt8] = data.bytes //[UInt8](data)
        // let motionDataset = try decoder.decode(MotionDataset.self, from: bytes)
        // print("Done...")
        // print(motionDataset.describe())
        // print(abs(date.timeIntervalSinceNow))
    }

    static var allTests = [
        // ("serializeMotionFrame", serializeMotionFrame),
        ("serializeMotionSample", serializeMotionSample),
        ("serializeMotionDatasetJSON", serializeMotionDatasetJSON),
        ("serializeMotionDatasetBinary", serializeMotionDatasetBinary),
        ("readBinaryMotionDataset", readBinaryMotionDataset),
    ]
}
