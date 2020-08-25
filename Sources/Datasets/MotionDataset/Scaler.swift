import TensorFlow


public struct Scaler {
    // https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    public var mean: Tensor<Float>? = nil
    public var scale: Tensor<Float>? = nil

    public init() {
    }

    public init(X: Tensor<Float>) {
        fit(X)
    }

    public init(mean: Tensor<Float>, scale: Tensor<Float>) {
        self.mean = mean
        self.scale = scale
    }

    public mutating func fit(_ X: Tensor<Float>) {
        mean = X.mean(squeezingAxes:0)
        scale = sqrt(X.variance(squeezingAxes:0))
    }

    public func transform(_ X: Tensor<Float>) -> Tensor<Float> {
        var _X = X
        _X -= mean!
        _X /= scale!
        return _X
    }

    public func inverse_transform(_ X: Tensor<Float>) -> Tensor<Float> {
        return (X .* scale! + mean!)
    }
}
