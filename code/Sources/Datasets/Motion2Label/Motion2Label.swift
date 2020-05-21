import Foundation
import TensorFlow
import Batcher
import PythonKit


let pd = Python.import("pandas")


extension Tensor where Scalar: Numeric {
    func paddedOrCropped(to width: Int) -> Tensor<Scalar> {
        // pads or crops two-dimensional tensor along 0-th axis
        assert(self.shape.count == 2)
        let currentWidth = self.shape[0]
        let nPadding = Swift.max(width - currentWidth, 0)
        let maxCropping = Swift.max(currentWidth - width, 0)
        let nCropping = (maxCropping>0) ? Int.random(in: 0 ..< maxCropping) : 0
        return self[nCropping..<nCropping+width].padded(forSizes: [(before: 0, after: nPadding), (before: 0, after: 0)])
    }
}


func getTensorLabel(_ ms: MotionSample, labelsDict: [Int: String], labels: [String], tensorWidth: Int) -> (Tensor<Float>, Int32) {
    // TODO: code _unknown_ label
    var labelStr = labelsDict[ms.sampleID]
    
    if labelStr == nil {
        print("FIXME: unknown label")
        labelStr = "Doing something"
    }
    
    let label: Int32 = Int32(labels.firstIndex(of: labelStr!)!)
  
    var tensor = Tensor<Float>(ms.motionFramesArray)
    tensor = tensor.paddedOrCropped(to: tensorWidth).expandingShape(at: 2)
    return (tensor, label)
}


extension Array { 
    func trainTestSplit(split: Float) -> (train: Array<Element>, test: Array<Element>) {
        let shuffled = self.shuffled()
        let splitIdx = Int(roundf(Float(split * Float(self.count))))
        let train = Array(shuffled[0..<splitIdx])
        let test = Array(shuffled[splitIdx..<self.count])
        return (train: train, test: test)
    }
}


public struct Motion2Label {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public let training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>
    public let labels: [String]
    public let motionData: MotionData

    public init(batchSize: Int, serializedDatasetURL: URL, labelsURL: URL) {
        let motionData2 = MotionData(from: serializedDatasetURL)
        print(motionData2.description)
        
        let df = pd.read_csv(labelsURL.path)
        let labels2 = df.label.unique().sorted().map {String($0)!}

        var labelsDict: [Int: String] = [:]
        for pythonTuple in df.iterrows() {
            labelsDict[Int(pythonTuple[1].sample_id)!] = String(pythonTuple[1].label)!
        }
        
        // filter out samples without annotations
        let motionSamples = motionData2.motionSamples.filter { $0.annotations.count > 0 }
        
        let tensorPairs: [(Tensor<Float>, Int32)] = motionSamples.map { 
            getTensorLabel($0, labelsDict: labelsDict, labels: labels2, tensorWidth: 600) 
        }
        print("tensorPairs.count: \(tensorPairs.count)")
        let tensorPairs2: SourceDataSet = tensorPairs.map { TensorPair(first: $0.0, second: Tensor<Int32>($0.1)) }
        let (trainTensorPairs, testTensorPairs) = tensorPairs2.trainTestSplit(split: 0.8)
        
        print("trainTensorPairs.count = \(trainTensorPairs.count)")
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
            shuffle: false)
        self.labels = labels2
        self.motionData = motionData2
    }    
}
