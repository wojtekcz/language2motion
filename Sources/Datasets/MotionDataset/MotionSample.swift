import Foundation
import TensorFlow

public struct MotionSample: Codable {
    public let sampleID: Int
    public let jointNames: [String]
    public let annotations: [String]

    public let timesteps: Tensor<Float> // 1D, time steps
    public var motion: Tensor<Float> // 2D, [motion frames x joint positions], without motion flag
    public var discreteMotion: Tensor<Int32>? = nil // motion frames x joint positions

    enum CodingKeys: String, CodingKey {
        case sampleID
        case jointNames
        case annotations
        case timesteps
        case motion
    }

    public init(sampleID: Int, annotations: [String], jointNames: [String], timesteps: Tensor<Float>, motion: Tensor<Float>) {
        self.sampleID = sampleID
        self.jointNames = jointNames
        self.annotations = annotations
        self.timesteps = timesteps
        self.motion = motion
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        sampleID = try values.decode(Int.self, forKey: .sampleID)
        jointNames = try values.decode(Array<String>.self, forKey: .jointNames)
        annotations = try values.decode(Array<String>.self, forKey: .annotations)
        timesteps = try! values.decode(FastCodableTensor<Float>.self, forKey: .timesteps).tensor
        motion = try! values.decode(FastCodableTensor<Float>.self, forKey: .motion).tensor
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sampleID, forKey: .sampleID)
        try container.encode(jointNames, forKey: .jointNames)
        try container.encode(annotations, forKey: .annotations)
        try container.encode(FastCodableTensor<Float>(timesteps), forKey: .timesteps)
        try container.encode(FastCodableTensor<Float>(motion), forKey: .motion)
    }

    public var description: String {
        return "MotionSample(timesteps: \(timesteps[-1].scalar!), motion: \(motion.shape[0]), annotations: \(annotations.count))"
    }
}

extension MotionSample {
    public static func grouppedJoints(motion: Tensor<Float>, jointNames: [String]) -> Tensor<Float> {
        let gIdxs = MotionFrame.grouppedJointPositionIdxs(jointNames: jointNames)
        return Tensor<Float>(stacking: (0..<gIdxs.count).map { motion[0..., gIdxs[$0]] }, alongAxis: 1)
    }

    public func grouppedJoints(motion: Tensor<Float>) -> Tensor<Float> {
        return Self.grouppedJoints(motion: motion, jointNames: self.jointNames)
    }
}
