import Foundation
import FoundationXML
import TensorFlow
import PythonKit

let np = Python.import("numpy")

// TODO: move to some utils
func tensorFromArray(arr: Array<Array<Float>>) -> Tensor<Float>? {
    // TODO: convert to extension with custom init
    return Tensor<Float>.init(numpy: np.array(arr).astype(np.float32))
}
func shapedArrayFromArray(arr: Array<Array<Float>>) -> ShapedArray<Float>? {
    // TODO: convert to extension with custom init
    return ShapedArray<Float>.init(numpy: np.array(arr).astype(np.float32))
}


public struct MotionSample: Codable {
    public let sampleID: Int
    public let motionFrames: [MotionFrame]
    public let jointNames: [String]
    public let annotations: [String]

    public let timestamps: [Float]
    public let motionFramesArray: ShapedArray<Float>

    enum CodingKeys: String, CodingKey {
        case sampleID
        case jointNames
        case annotations
        case timestamps
        case motionFramesArray
    }

    public init(sampleID: Int, mmmURL: URL, annotationsURL: URL) {
        self.sampleID = sampleID
        let mmm_doc = MotionSample.loadMMM(fileURL: mmmURL)
        let jointNames = MotionSample.getJointNames(mmm_doc: mmm_doc)
        self.jointNames = jointNames
        let motionFrames = MotionSample.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)
        self.motionFrames = motionFrames
        self.annotations = MotionSample.getAnnotations(fileURL: annotationsURL)
        self.timestamps = motionFrames.map { $0.timestamp }
        self.motionFramesArray = MotionSample.getJointPositions(motionFrames: motionFrames, grouppedJoints: false, normalized: false)
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        sampleID = try values.decode(Int.self, forKey: .sampleID)
        jointNames = try values.decode(Array<String>.self, forKey: .jointNames)
        annotations = try values.decode(Array<String>.self, forKey: .annotations)
        timestamps = try values.decode(Array<Float>.self, forKey: .timestamps)
        motionFramesArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .motionFramesArray).shapedArray

        // loop over motionFramesArray and create MotionFrames
        var motionFrames: [MotionFrame] = []
        for f_idx in 0..<motionFramesArray.shape[0] {
            let mf = MotionFrame(
                timestamp: timestamps[f_idx], 
                jointPositions: motionFramesArray[f_idx].scalars, 
                jointNames: jointNames
            )
            motionFrames.append(mf)
        }
        self.motionFrames = motionFrames
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sampleID, forKey: .sampleID)
        try container.encode(jointNames, forKey: .jointNames)
        try container.encode(annotations, forKey: .annotations)
        try container.encode(timestamps, forKey: .timestamps)
        try container.encode(FastCodableShapedArray<Float>(shapedArray: motionFramesArray), forKey: .motionFramesArray)
    }

    static func loadMMM(fileURL: URL) -> XMLDocument {
        let mmm_text = try! String(contentsOf: fileURL, encoding: .utf8)
        return try! XMLDocument(data: mmm_text.data(using: .utf8)!, options: [])
    }
    
    static func getAnnotations(fileURL: URL) -> [String] {
        let annotationsData = try! Data(contentsOf: fileURL)
        return try! JSONSerialization.jsonObject(with: annotationsData) as! [String]        
    }
    
    static func getJointNames(mmm_doc: XMLDocument) -> [String] {
        let jointNode: [XMLNode] = try! mmm_doc.nodes(forXPath: "/MMM/Motion/JointOrder/Joint/@name")
        return jointNode.map {$0.stringValue!.replacingOccurrences(of: "_joint", with: "")}
    }
    
    static func getMotionFrames(mmm_doc: XMLDocument, jointNames: [String]) -> [MotionFrame] {
        var motionFrames: [MotionFrame] = []
        var count = 0
        for motionFrame in try! mmm_doc.nodes(forXPath: "/MMM/Motion/MotionFrames/MotionFrame") {
            count += 1
            
            let timestampStr: String = (try! motionFrame.nodes(forXPath:"Timestep"))[0].stringValue!
            let jointPositionStr: String = (try! motionFrame.nodes(forXPath:"JointPosition"))[0].stringValue!
            let jointPositions: [Float] = jointPositionStr.split(separator: " ").map {
                var xx = Float($0)
                if xx==nil { xx = 0.0 }
                return xx!
            }

            let mf = MotionFrame(
                timestamp: Float(timestampStr)!, 
                jointPositions: jointPositions, 
                jointNames: jointNames
            )
            motionFrames.append(mf)
        }
        return motionFrames
    }

    static func getJointPositions(motionFrames: [MotionFrame], grouppedJoints: Bool, normalized: Bool) -> ShapedArray<Float> {
        var a: Array<Array<Float>>? = nil
        if grouppedJoints {
            a = motionFrames.map {$0.grouppedJointPositions()}
        } else {
            a = motionFrames.map {$0.jointPositions}
        }
        if normalized {
            let t = sigmoid(tensorFromArray(arr: a!)!)
            return t.array
        } else {
            return shapedArrayFromArray(arr: a!)!
        }
    }

    public var description: String {
        return "MotionSample(timestamp: \(self.motionFrames.last!.timestamp), motions: \(self.motionFrames.count), annotations: \(self.annotations.count))"
    }
}
