import TensorFlow


public struct MinMaxScaler {
    // https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    public var min: Tensor<Float>? = nil
    public var max: Tensor<Float>? = nil

    public init() {
    }

    public init(X: Tensor<Float>) {
        fit(X)
    }

    public init(min: Tensor<Float>, max: Tensor<Float>) {
        self.min = min
        self.max = max
    }

    public mutating func fit(_ X: Tensor<Float>) {
        min = X.min(alongAxes: 0)
        max = X.max(alongAxes: 0)
    }

    public func transform(_ X: Tensor<Float>) -> Tensor<Float> {
        // X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        let X_std = (X - self.min!) / (self.max! - self.min!)
        return X_std
    }

    public func inverse_transform(_ X: Tensor<Float>) -> Tensor<Float> {
        // X_scaled = X_std * (max - min) + min
        let X_std = X
        let X_scaled = X_std * (self.max! - self.min!) + self.min!
        return X_scaled
    }
}
