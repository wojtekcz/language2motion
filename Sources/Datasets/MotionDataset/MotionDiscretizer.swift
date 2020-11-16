import Foundation
import TensorFlow
import PythonKit


public struct MotionDiscretizer: Codable {
    // https://towardsdatascience.com/an-introduction-to-discretization-in-data-science-55ef8c9775a2
    let preprocessing = Python.import("sklearn.preprocessing")
    public let discretizer: PythonObject
    public let strategy: String
    public let encode: String
    public let n_bins: Int

    public static func createFromJSONURL(_ url: URL) -> Self? {
        let json = try! String(contentsOf: url, encoding: .utf8).data(using: .utf8)!
        guard let f = try? JSONDecoder().decode(MotionDiscretizer.self, from: json) else {
            return nil
        }
        return f
    }

    public init(n_bins: Int = 300, X: Tensor<Float>) {
        self.n_bins = n_bins
        self.strategy = "uniform" // "quantile"
        self.encode = "ordinal"
        discretizer = preprocessing.KBinsDiscretizer(n_bins: n_bins, encode: encode, strategy: strategy)
        fit(X)
    }

    public init(n_bins: Int = 300) {
        self.n_bins = n_bins
        self.strategy = "uniform" // "quantile"
        self.encode = "ordinal"
        discretizer = preprocessing.KBinsDiscretizer(n_bins: n_bins, encode: encode, strategy: strategy)
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

    enum CodingKeys: String, CodingKey {
        case strategy
        case encode
        case n_bins
        case n_bins_
        case bin_edges_
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        encode = try values.decode(String.self, forKey: .encode)
        strategy = try values.decode(String.self, forKey: .strategy)
        n_bins = try values.decode(Int.self, forKey: .n_bins)
        let n_bins_ = try values.decode(Array<Int32>.self, forKey: .n_bins_)
        let bin_edges_: [[Double]] = try values.decode(Array.self, forKey: .bin_edges_)

        discretizer = preprocessing.KBinsDiscretizer(n_bins: n_bins, encode: "ordinal", strategy: strategy)
        discretizer.n_bins_ = n_bins_.makeNumpyArray()
        discretizer.bin_edges_ = [bin_edges_[0].makeNumpyArray()]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(strategy, forKey: .strategy)
        try container.encode(encode, forKey: .encode)
        try container.encode(n_bins, forKey: .n_bins)

        let n_bins_: [Int32] = Array(discretizer.n_bins_)!
        try container.encode(n_bins_, forKey: .n_bins_)

        let bin_edges_: [[Double]] = Array(discretizer.bin_edges_)!
        try container.encode(bin_edges_, forKey: .bin_edges_)
    }
}
