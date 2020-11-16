import Foundation
import XCTest
import TensorFlow
import PythonKit
import Datasets


class MotionDiscretizerTests: XCTestCase {

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func testDiscretizerWriting() throws {
        let dsMgr = DatasetManager(datasetSize: .small_micro1, device: Device.defaultTFEager)
        let dataset = dsMgr.dataset!
        print("dataset: \(dataset.motionSamples.count)")
        
        let n_bins = 300
        
        let discretizer = dataset.discretizer
        let params = discretizer.discretizer.get_params(true)
        print("params: \(params)")

        let python_n_bins_ = discretizer.discretizer.n_bins_
        print("Python.type(python_n_bins_): \(Python.type(python_n_bins_))")
        print("python_n_bins_.shape: \(python_n_bins_.shape)")
        
        let python_bin_edges_ = discretizer.discretizer.bin_edges_
        print("python_bin_edges_.shape: \(python_bin_edges_.shape)")
//        print("python_bin_edges_: \(python_bin_edges_)")

//        let n_bins_: [Int] = Array(discretizer.discretizer.n_bins_)!
//        print("n_bins_: \(n_bins_)")

//        let bin_edges_: [[Double]] = Array(discretizer.discretizer.bin_edges_)!
//        print("bin_edges_: \(bin_edges_)")

        if let encodedData = try? JSONEncoder().encode(discretizer) {
            let discretizerURL = dataURL.appendingPathComponent("motion_discretizer.\(n_bins).json")
            do {
                try encodedData.write(to: discretizerURL)
            }
            catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
        
        // test discrtetizer transform
        let t1 = Tensor<Float>(randomNormal: [10, 47])
        let _ = discretizer.transform(t1)
//        print(t2)
        print()
    }

    func testDiscretizerReading() throws {
        let discretizerURL = dataURL.appendingPathComponent("motion_discretizer.300.json")
        
        let json = try! String(contentsOf: discretizerURL, encoding: .utf8).data(using: .utf8)!
        let discretizer = try! JSONDecoder().decode(MotionDiscretizer.self, from: json)
        print("discretizer: \(discretizer)")

//        let n_bins_: [Int] = Array(discretizer.discretizer.n_bins_)!
//        print("n_bins_: \(n_bins_)")

//        let bin_edges_: [[Double]] = [Array(discretizer.discretizer.bin_edges_[0])!]
//        print("bin_edges_: \(bin_edges_)")
        
        // test discrtetizer transform and inverse transform
        let t1 = Tensor<Float>(randomNormal: [10, 47])
        let t2 = discretizer.transform(t1)
        //print(t2)
        let _ = discretizer.inverse_transform(t2)
        //print(t3)
        print()
    }
}
