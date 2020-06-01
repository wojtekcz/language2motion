import TensorFlow
import PythonKit


let np1 = Python.import("numpy")


extension Array where Element == Array<Float> {
    func makeTensor() -> Tensor<Float> {
        return Tensor<Float>.init(numpy: np1.array(self).astype(np1.float32))!
    }
    func makeShapedArray() -> ShapedArray<Float> {
        return ShapedArray<Float>.init(numpy: np1.array(self).astype(np1.float32))!
    }
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
