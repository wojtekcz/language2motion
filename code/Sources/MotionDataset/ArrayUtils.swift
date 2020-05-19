import TensorFlow
import PythonKit


let np = Python.import("numpy")


extension Array where Element == Array<Float> {
    func makeTensor() -> Tensor<Float> {
        return Tensor<Float>.init(numpy: np.array(self).astype(np.float32))!
    }
    func makeShapedArray() -> ShapedArray<Float> {
        return ShapedArray<Float>.init(numpy: np.array(self).astype(np.float32))!
    }
}
