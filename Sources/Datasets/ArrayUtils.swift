import TensorFlow
import PythonKit


let np1 = Python.import("numpy")


extension Array where Element == Array<Float> {
    public func makeTensor() -> Tensor<Float> {
        return Tensor<Float>.init(numpy: np1.array(self).astype(np1.float32))!
    }
    public func makeShapedArray() -> ShapedArray<Float> {
        return ShapedArray<Float>.init(numpy: np1.array(self).astype(np1.float32))!
    }
}

extension Array { 
    public func trainTestSplit(split: Double) -> (train: Array<Element>, test: Array<Element>) {
        let shuffled = self.shuffled()
        let splitIdx = Int(roundf(Float(split * Double(self.count))))
        let train = Array(shuffled[0..<splitIdx])
        let test = Array(shuffled[splitIdx..<self.count])
        return (train: train, test: test)
    }
}

extension Collection {
    public func choose(_ n: Int) -> ArraySlice<Element> { shuffled().prefix(n) }
}

extension String { 
    public func floatArray() -> [Float] {
        self.split(separator: " ").map {
                var value = Float($0)
                if value==nil { value = 0.0 }
                return value!
            }
    }
}
