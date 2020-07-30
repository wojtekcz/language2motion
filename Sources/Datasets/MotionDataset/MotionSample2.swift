import Foundation
import TensorFlow

public struct MotionSample2: Codable {
    public let sampleID: Int
    public let jointNames: [String]
    public let annotations: [String]

    // TODO: change ShapedArrays to Tensors
    public let timestepsArray: ShapedArray<Float> // 1D, time steps
    public let motionFramesArray: ShapedArray<Float> // 2D, [motion frames x joint positions], without motion flag

    enum CodingKeys: String, CodingKey {
        case sampleID
        case jointNames
        case annotations
        case timestepsArray
        case motionFramesArray
    }

    public init(sampleID: Int, annotations: [String], jointNames: [String], timestepsArray: ShapedArray<Float>, motionFramesArray: ShapedArray<Float>) {
        self.sampleID = sampleID
        self.jointNames = jointNames
        self.annotations = annotations
        self.timestepsArray = timestepsArray
        self.motionFramesArray = motionFramesArray
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        sampleID = try values.decode(Int.self, forKey: .sampleID)
        jointNames = try values.decode(Array<String>.self, forKey: .jointNames)
        annotations = try values.decode(Array<String>.self, forKey: .annotations)
        timestepsArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .timestepsArray).shapedArray
        motionFramesArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .motionFramesArray).shapedArray
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sampleID, forKey: .sampleID)
        try container.encode(jointNames, forKey: .jointNames)
        try container.encode(annotations, forKey: .annotations)
        try container.encode(FastCodableShapedArray<Float>(shapedArray: timestepsArray), forKey: .timestepsArray)
        try container.encode(FastCodableShapedArray<Float>(shapedArray: motionFramesArray), forKey: .motionFramesArray)
    }

    public var description: String {
        return "MotionSample2(timesteps: \(timestepsArray[-1].scalar!), motion: \(motionFramesArray.shape[0]), annotations: \(annotations.count))"
    }
}
