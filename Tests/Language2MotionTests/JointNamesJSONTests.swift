import Foundation
import XCTest
import TensorFlow
import PythonKit
import Datasets


class JointNamesJSONTests: XCTestCase {

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func testWriteJointNames() throws {
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: Device.defaultTFEager)

        let jointNames = dsMgr.dataset!.motionSamples[0].jointNames

        if let encodedData = try? JSONEncoder().encode(jointNames) {
            let jointNamesURL = dataURL.appendingPathComponent("joint_names.json")
            do {
                try encodedData.write(to: jointNamesURL)
            }
            catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
    }

    func testJointNamesReading() throws {
        let json = try! String(contentsOf: dataURL.appendingPathComponent("joint_names.json"), encoding: .utf8).data(using: .utf8)!
        let jointNames = try! JSONDecoder().decode(Array<String>.self, from: json)
        print("jointNames: \(jointNames)")
    }
}
