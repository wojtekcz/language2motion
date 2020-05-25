import Foundation
import FoundationXML
import TensorFlow


public struct MotionSample: Codable {
    public let sampleID: Int
    public let motionFrames: [MotionFrame]
    public let jointNames: [String]
    public let annotations: [String]

    public let timestampsArray: ShapedArray<Float> // 1D, time steps
    public let motionFramesArray: ShapedArray<Float> // 2D, motion frames, joint positions

    enum CodingKeys: String, CodingKey {
        case sampleID
        case jointNames
        case annotations
        case timestampsArray
        case motionFramesArray
    }

    public init(sampleID: Int, mmmURL: URL, annotationsURL: URL, grouppedJoints: Bool = true, normalized: Bool = true) {
        self.sampleID = sampleID
        let mmm_doc = MotionSample.loadMMM(fileURL: mmmURL)
        let jointNames = MotionSample.getJointNames(mmm_doc: mmm_doc)
        self.jointNames = jointNames
        let motionFrames = MotionSample.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)
        self.motionFrames = motionFrames
        self.annotations = MotionSample.getAnnotations(fileURL: annotationsURL)
        let timestamps: [Float] = motionFrames.map { $0.timestamp }
        self.timestampsArray = ShapedArray<Float>(shape: [timestamps.count], scalars: timestamps)
        self.motionFramesArray = MotionSample.getJointPositions(motionFrames: motionFrames, grouppedJoints: grouppedJoints, normalized: normalized)
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        sampleID = try values.decode(Int.self, forKey: .sampleID)
        jointNames = try values.decode(Array<String>.self, forKey: .jointNames)
        annotations = try values.decode(Array<String>.self, forKey: .annotations)
        timestampsArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .timestampsArray).shapedArray
        motionFramesArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .motionFramesArray).shapedArray

        // loop over motionFramesArray and create MotionFrames
        var motionFrames: [MotionFrame] = []
        let timestamps = timestampsArray.scalars // working with scalars, for performance
        let mfScalars = motionFramesArray.scalars // working with scalars, for performance
        let cj = motionFramesArray.shape[1]
        for i in 0..<motionFramesArray.shape[0] {
            let start = i*cj
            let jointPositions: [Float] = Array(mfScalars[start..<(start+cj)])
            let mf = MotionFrame(
                timestamp: timestamps[i], 
                jointPositions: jointPositions, 
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
        try container.encode(FastCodableShapedArray<Float>(shapedArray: timestampsArray), forKey: .timestampsArray)
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
            let t = sigmoid(a!.makeTensor())
            return t.array
        } else {
            return a!.makeShapedArray()
        }
    }

    public var description: String {
        return "MotionSample(timestamp: \(self.motionFrames.last!.timestamp), motions: \(self.motionFrames.count), annotations: \(self.annotations.count))"
    }
}
