import Foundation
import TensorFlow
import PythonKit


public struct MotionDiscretizer {
    // https://towardsdatascience.com/an-introduction-to-discretization-in-data-science-55ef8c9775a2
    let preprocessing = Python.import("sklearn.preprocessing")
    let discretizer: PythonObject
    let strategy = "uniform" // "quantile"

    public init(n_bins: Int = 300, X: Tensor<Float>) {
        discretizer = preprocessing.KBinsDiscretizer(n_bins: n_bins, encode: "ordinal", strategy: strategy)
        fit(X)
    }

    public mutating func fit(_ X: Tensor<Float>) {
        discretizer.fit(X.flattened().expandingShape(at: 1).makeNumpyArray())
    }

    public func transform(_ X: Tensor<Float>) -> Tensor<Int32> {
        let t_np = discretizer.transform(X.flattened().expandingShape(at: 1).makeNumpyArray())
        return Tensor<Int32>(Tensor<Float>(numpy: t_np)!.reshaped(like: X))
    }

    public func inverse_transform(_ X: Tensor<Int32>) -> Tensor<Float> {
        let t_np = discretizer.inverse_transform(X.flattened().expandingShape(at: 1).makeNumpyArray())
        return Tensor<Float>(Tensor<Double>(numpy: t_np)!.reshaped(like: X))
    }
}
