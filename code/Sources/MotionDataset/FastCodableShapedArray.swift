import Foundation
import TensorFlow

extension Array {
    // https://stackoverflow.com/questions/49555088/swift-best-way-to-send-large-arrays-of-numbers-double-over-http

    init?(data: Data) {
        // This check should be more complex, but here we just check if total byte count divides to one element size in bytes
        guard data.count % MemoryLayout<Element>.size == 0 else { return nil }

        let elementCount = data.count / MemoryLayout<Element>.size
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: elementCount)
        let _ = data.copyBytes(to: buffer)

        self = buffer.map({$0})
        buffer.deallocate()
    }

   // Wrapped here code above
    var data: Data {
        return self.withUnsafeBufferPointer { pointer in
            return Data(buffer: pointer)
        }
    }
}

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
