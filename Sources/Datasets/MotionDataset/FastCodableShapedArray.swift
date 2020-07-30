import Foundation
import TensorFlow


public struct FastCodableShapedArray<Scalar>: Codable {
    // Fast Codable ShapedArray
    // encoding Scalars as Data
    let shapedArray: ShapedArray<Scalar>
    
    public init (shapedArray: ShapedArray<Scalar>) {
        self.shapedArray = shapedArray
    }
    
    private enum CodingKeys: String, CodingKey {
        case shape
        case scalarsData
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let shape = try container.decode([Int].self, forKey: .shape)
        let data = try container.decode(Data.self, forKey: .scalarsData)
        let scalars: [Scalar] = Array(data: data)!
        shapedArray = ShapedArray(shape: shape, scalars: scalars)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(shapedArray.shape, forKey: .shape)
        let data = shapedArray.scalars.data
        try container.encode(data, forKey: .scalarsData)
    }
}
