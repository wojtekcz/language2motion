import Foundation
import TensorFlow
import Batcher
import PythonKit


public class Motion2Label {
    public typealias SourceDataSet = [TensorPair<Float, Int32>]
    public var training: Batcher<SourceDataSet>
    public let test: Batcher<SourceDataSet>
    public let labels: [String]
    public let motionData: MotionData
    public let trainMotionSamples: [MotionSample]
    public let testMotionSamples: [MotionSample]
    public let batchSize: Int
    public let tensorWidth: Int
    public let labelsDict: [Int: String]

    public init(batchSize: Int, serializedDatasetURL: URL, labelsURL: URL, tensorWidth: Int = 600) {
        let motionData2 = MotionData(from: serializedDatasetURL)
        print(motionData2.description)
        
        let df = pd.read_csv(labelsURL.path)
        let labels2 = df.label.unique().sorted().map {String($0)!}

        var labelsDict2: [Int: String] = [:]
        for pythonTuple in df.iterrows() {
            labelsDict2[Int(pythonTuple[1].sample_id)!] = String(pythonTuple[1].label)!
        }
        // filter out samples without annotations
        let motionSamples = motionData2.motionSamples.filter { $0.annotations.count > 0 }
        
        // split into train/test sets
        (trainMotionSamples, testMotionSamples) = motionSamples.trainTestSplit(split: 0.8)

        // samples to tensors
        let trainTensorPairs: SourceDataSet = trainMotionSamples.map {
            Motion2Label.getTensorPair($0, labelsDict: labelsDict2, labels: labels2, tensorWidth: tensorWidth) 
        }
        let testTensorPairs: SourceDataSet = testMotionSamples.map {
            Motion2Label.getTensorPair($0, labelsDict: labelsDict2, labels: labels2, tensorWidth: tensorWidth)
        }
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
        self.batchSize = batchSize
        self.labelsDict = labelsDict2
        self.tensorWidth = tensorWidth
    }

    public static func getTensorPair(_ ms: MotionSample, labelsDict: [Int: String], labels: [String], tensorWidth: Int) -> TensorPair<Float, Int32> {
        // TODO: code _unknown_ label
        var labelStr = labelsDict[ms.sampleID]
        
        if labelStr == nil {
            print("FIXME: unknown label")
            labelStr = "Doing something"
        }
        
        let label: Tensor<Int32> = Tensor<Int32>(Int32(labels.firstIndex(of: labelStr!)!))
    
        var tensor = Tensor<Float>(ms.motionFramesArray)
        // FIXME: make getJointPositions much faster
        // no need, following transformation is done during dataset preprocessing
        // var tensor = Tensor<Float>(MotionSample.getJointPositions(motionFrames: ms.motionFrames, grouppedJoints: true, normalized: true))
        tensor = tensor.paddedOrCropped(to: tensorWidth).expandingShape(at: 2)
        return TensorPair(first: tensor, second: label)
    }

    public func newTrainCrops() {
        let trainTensorPairs: SourceDataSet = trainMotionSamples.map {
            Motion2Label.getTensorPair($0, labelsDict: self.labelsDict, labels: self.labels, tensorWidth: self.tensorWidth) 
        }
        self.training = Batcher(
            on: trainTensorPairs,
            batchSize: batchSize,
            numWorkers: 1, //No need to use parallelism since everything is loaded in memory
            shuffle: true)
    }

    static func getLabel(sampleID: Int, labelsDict: [Int: String], labels: [String]) -> Motion2LabelExample.LabelTuple? {
        let labelStr: String? = labelsDict[sampleID]
        
        var label: Motion2LabelExample.LabelTuple? = nil
        if labelStr != nil {
            label = Motion2LabelExample.LabelTuple(idx: labels.firstIndex(of: labelStr!)!, label: labelStr!)
        }
        return label
    }

    public func getLabel(_ sampleID: Int) -> Motion2LabelExample.LabelTuple? {
        return Motion2Label.getLabel(sampleID: sampleID, labelsDict: labelsDict, labels: labels)
    }

}
