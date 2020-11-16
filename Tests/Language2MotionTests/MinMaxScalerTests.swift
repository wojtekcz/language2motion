import Foundation
import XCTest
import TensorFlow
import PythonKit
import Datasets


class MinMaxScalerTests: XCTestCase {

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func testScalerWriting() throws {
        let datasetSize: DatasetSize = .full
        let dsMgr = DatasetManager(datasetSize: datasetSize, device: Device.defaultTFEager)
        let dataset = dsMgr.dataset!
        print("dataset: \(dataset.motionSamples.count)")
        
        let scaler = dataset.scaler
        
        if let encodedData = try? JSONEncoder().encode(scaler) {
            let scalerURL = dataURL.appendingPathComponent("min_max_scaler.\(datasetSize.rawValue)json")
            do {
                try encodedData.write(to: scalerURL)
            }
            catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
    }

    func testScalerReading() throws {
        let scalerURL = dataURL.appendingPathComponent("min_max_scaler.json")
        
        let json = try! String(contentsOf: scalerURL, encoding: .utf8).data(using: .utf8)!
        let scaler = try! JSONDecoder().decode(MinMaxScaler.self, from: json)
        print("scaler: \(scaler)")
    }
}
