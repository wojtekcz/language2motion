import Foundation
import TensorFlow


public struct FastCodableTensor<Scalar>: Codable where Scalar: TensorFlowScalar {
    // Fast implementation of Codable Tensor
    // encoding scalars as Data object
    public let tensor: Tensor<Scalar>
    
    public init (_ tensor: Tensor<Scalar>) {
        self.tensor = tensor
    }
    
    private enum CodingKeys: String, CodingKey {
        case shape
        case scalarsData
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try container.decode(TensorShape.self, forKey: .shape)
        let data = try container.decode(Data.self, forKey: .scalarsData)
        let scalars: [Scalar] = Array(data: data)!
        tensor = Tensor<Scalar>(shape: shape, scalars: scalars)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(tensor.shape, forKey: .shape)
        let data = tensor.scalars.data
        try container.encode(data, forKey: .scalarsData)
    }
}
