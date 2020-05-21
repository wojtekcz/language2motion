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
