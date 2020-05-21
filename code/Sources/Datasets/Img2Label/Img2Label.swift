// TODO:
// - download dataset if not present
// - unpack dataset
// + enumerate folders, get (sorted) labels
// + enumerate images
// + sort image paths
// + load image
// + create [TensorPair<Float, Int32>] list
// + create dataset object
// + calculate normalization
// + split into training/test


import Foundation
import PythonKit
import ModelSupport
import TensorFlow
import Batcher


let np  = Python.import("numpy")
let sklearn  = Python.import("sklearn")
let model_selection  = Python.import("sklearn.model_selection")
let glob = Python.import("glob")
let Image = Python.import("PIL.Image")


public struct Img2Label: ImageClassificationDataset {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>
    public let labels: [String]

    public init(batchSize: Int) {
        let dsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/img2label_ds_v1", isDirectory: true)
        self.init(batchSize: batchSize, dsURL: dsURL)
    }

    public init(batchSize: Int, dsURL: URL, normalizing: Bool = true) {
        let labels2: [String] = Img2Label.loadLabels(dsURL)
        
        let imageList = glob.glob(dsURL.path + "/**/*.png")
        let (trainPythonList, testPythonList) = model_selection.train_test_split(imageList, test_size: 0.2).tuple2
        let (trainList, testList): ([String], [String]) = (Array(trainPythonList)!.sorted(), Array(testPythonList)!.sorted())
        
        let trainTensorPairs = loadImages(imageList: trainList, labels: labels2, normalizing: normalizing)
        print("trainTensorPairs.count = \(trainTensorPairs.count)")
        let testTensorPairs = loadImages(imageList: testList, labels: labels2, normalizing: normalizing)
        print("testTensorPairs.count = \(testTensorPairs.count)")
        self.training = Batcher(
            on: trainTensorPairs,
            batchSize: batchSize,
            numWorkers: 1, //No need to use parallelism since everything is loaded in memory
            shuffle: true)
        self.test = Batcher(
            on: testTensorPairs,
            batchSize: batchSize,
            numWorkers: 1,
            shuffle: true)
        self.labels = labels2
    }
    
    static func loadLabels(_ dsURL: URL) -> [String] {
        let fm = FileManager()
        let labels = try! fm.contentsOfDirectory(atPath: dsURL.path).filter({ !$0.hasPrefix(".") }).sorted()
        return labels
    }
}

/// load the image and extract the label
/// from Swift4TF_TransferLearning.ipynb
func getTensorLabel(_ imageURL: URL, labels: [String]) -> (Tensor<Float>, Int32) {
    let label: Int32
  
    let labelStr = imageURL.deletingLastPathComponent().lastPathComponent
    label = Int32(labels.firstIndex(of: labelStr)!)
    let img = Image.open(imageURL.path)
    let arr = np.array(img)[0..<224, 0..<224, 0..<3] // kill transparency
    let image = Tensor<UInt8>(numpy: arr)!

    var tensor = Tensor<Float>(image)
    tensor = _Raw.expandDims(tensor, dim: Tensor<Int32>(0))
    tensor = _Raw.resizeArea(images:tensor, size:[224, 224])
    return (tensor.squeezingShape(at: 0), label)
}

func loadImages(imageList: [String], labels: [String], normalizing: Bool = true) -> [TensorPair<Float, Int32>] {
    let tensorLabels = imageList.map { (path: String) -> (Tensor<Float>, Int32) in 
        return getTensorLabel(URL(fileURLWithPath: path), labels: labels)
    }
    
    let imageCount = tensorLabels.count
    let labelTensor = Tensor<Int64>(shape: [imageCount], scalars: tensorLabels.map {Int64($0.1)})

    var imageTensor = Tensor<Float>(stacking: tensorLabels.map {$0.0}, alongAxis: 0)

    // The value of mean and std were calculated with the following Swift code:
    // ```
    // import TensorFlow
    // import Datasets
    // import Foundation

    // let dsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/img2label_ds_v1", isDirectory: true)
    // let dataset = Img2Label(batchSize: 2500, dsURL: dsURL, normalizing: false)
    // print("dataset.training.count: \(dataset.training.count)")
    // for batch in dataset.training.sequenced() {
    //     let images = Tensor<Double>(batch.first) / 255.0
    //     let mom = images.moments(squeezingAxes: [0,1,2])
    //     print("mean: \(mom.mean) std: \(sqrt(mom.variance))")
    // }
    // ```
    if normalizing {
        let mean = Tensor<Float>(
                [0.8836673330105219,
                 0.8571306618582774,
                 0.5989467475049005])
        let std = Tensor<Float>(
                [0.1870305997172803,
                 0.1698038429051249,
                 0.11811759458558127])
        imageTensor = ((imageTensor / 255.0) - mean) / std
    }
    
    return (0..<imageCount).map { TensorPair(first: imageTensor[$0], second: Tensor<Int32>(labelTensor[$0])) }
        
}
