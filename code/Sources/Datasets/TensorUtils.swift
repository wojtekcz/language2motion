import TensorFlow

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
