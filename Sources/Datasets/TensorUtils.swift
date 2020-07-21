import TensorFlow

extension Tensor where Scalar: Numeric {
    public func paddedOrCropped(to width: Int) -> Tensor<Scalar> {
        // pads or crops one- or two-dimensional tensor along 0-th axis
        let currentWidth = self.shape[0]
        let maxCropping = Swift.max(currentWidth - width, 0)
        let nCropping = (maxCropping>0) ? Int.random(in: 0 ..< maxCropping) : 0
        return paddedAndCropped(to: width, nCropping: nCropping)
    }

    public func paddedAndCropped(to width: Int, nCropping: Int = 0) -> Tensor<Scalar> {
        // pads one- or two-dimensional tensor along 0-th axis
        let rank = self.shape.count
        let currentWidth = self.shape[0]
        let paddingSize = Swift.max(width - currentWidth, 0)
        var sizes: [(before: Int, after: Int)] = [(before: 0, after: paddingSize)]
        if rank > 1 {
            sizes.append((before: 0, after: 0))
        }
        return self[nCropping..<nCropping+width].padded(forSizes: sizes)
    }
}
