import Foundation
import FoundationXML
import TensorFlow


public struct MotionSample: Codable {
    public let sampleID: Int
    public let motionFrames: [MotionFrame]
    public let jointNames: [String]
    public let annotations: [String]

    public let timestepsArray: ShapedArray<Float> // 1D, time steps
    public let motionFramesArray: ShapedArray<Float> // 2D, motion frames, joint positions, motion flag

    enum CodingKeys: String, CodingKey {
        case sampleID
        case jointNames
        case annotations
        case timestepsArray
        case motionFramesArray
    }

    public init(sampleID: Int, mmmURL: URL, annotationsURL: URL, grouppedJoints: Bool = true, normalized: Bool = true, maxFrames: Int = 50000) {
        self.sampleID = sampleID
        let mmm_doc = MotionSample.loadMMM(fileURL: mmmURL)
        let jointNames = MotionSample.getJointNames(mmm_doc: mmm_doc)
        self.jointNames = jointNames
        var motionFrames = MotionSample.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)
        self.motionFrames = motionFrames
        self.annotations = MotionSample.getAnnotations(fileURL: annotationsURL)
        var timesteps: [Float] = motionFrames.map { $0.timestep }
        if motionFrames.count > maxFrames {
            motionFrames = Array(motionFrames[0..<maxFrames])
            timesteps = Array(timesteps[0..<maxFrames])
        }
        self.timestepsArray = ShapedArray<Float>(shape: [timesteps.count], scalars: timesteps)
        self.motionFramesArray = MotionSample.getJointPositions(motionFrames: motionFrames, grouppedJoints: grouppedJoints, normalized: normalized)
    }

    public init(sampleID: Int, motionFrames: [MotionFrame], annotations: [String], jointNames: [String], timesteps: [Float], grouppedJoints: Bool = true, normalized: Bool = true) {
        self.sampleID = sampleID
        self.jointNames = jointNames
        self.motionFrames = motionFrames
        self.annotations = annotations
        self.timestepsArray = ShapedArray<Float>(shape: [timesteps.count], scalars: timesteps)
        self.motionFramesArray = MotionSample.getJointPositions(motionFrames: motionFrames, grouppedJoints: grouppedJoints, normalized: normalized)
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        sampleID = try values.decode(Int.self, forKey: .sampleID)
        jointNames = try values.decode(Array<String>.self, forKey: .jointNames)
        annotations = try values.decode(Array<String>.self, forKey: .annotations)
        timestepsArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .timestepsArray).shapedArray
        motionFramesArray = try! values.decode(FastCodableShapedArray<Float>.self, forKey: .motionFramesArray).shapedArray

        // loop over motionFramesArray and create MotionFrames
        var motionFrames: [MotionFrame] = []
        let timesteps = timestepsArray.scalars // working with scalars, for performance
        let mfScalars = motionFramesArray.scalars // working with scalars, for performance
        let cj = motionFramesArray.shape[1]
        for i in 0..<motionFramesArray.shape[0] {
            let start = i*cj
            let jointPositions: [Float] = Array(mfScalars[start..<(start+cj)])
            let mf = MotionFrame(
                timestep: timesteps[i], 
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
        try container.encode(FastCodableShapedArray<Float>(shapedArray: timestepsArray), forKey: .timestepsArray)
        try container.encode(FastCodableShapedArray<Float>(shapedArray: motionFramesArray), forKey: .motionFramesArray)
    }

    public static func loadMMM(fileURL: URL) -> XMLDocument {
        let mmm_text = try! String(contentsOf: fileURL, encoding: .utf8)
        return try! XMLDocument(data: mmm_text.data(using: .utf8)!, options: [])
    }
    
    public static func getAnnotations(fileURL: URL) -> [String] {
        let annotationsData = try! Data(contentsOf: fileURL)
        return try! JSONSerialization.jsonObject(with: annotationsData) as! [String]        
    }
    
    public static func getJointNames(mmm_doc: XMLDocument) -> [String] {
        let jointNode: [XMLNode] = try! mmm_doc.nodes(forXPath: "/MMM/Motion/JointOrder/Joint/@name")
        return jointNode.map {$0.stringValue!.replacingOccurrences(of: "_joint", with: "")}
    }
    
    public static func getMotionFrames(mmm_doc: XMLDocument, jointNames: [String]) -> [MotionFrame] {
        var motionFrames: [MotionFrame] = []
        var count = 0
        for motionFrame in try! mmm_doc.nodes(forXPath: "/MMM/Motion/MotionFrames/MotionFrame") {
            count += 1
            
            let timestepStr: String = (try! motionFrame.nodes(forXPath:"Timestep"))[0].stringValue!
            let jointPositionStr: String = (try! motionFrame.nodes(forXPath:"JointPosition"))[0].stringValue!
            var jointPositions: [Float] = jointPositionStr.split(separator: " ").map {
                var value = Float($0)
                if value==nil { value = 0.0 }
                return value!
            }
            jointPositions += [1.0] // Adding motion flag

            let mf = MotionFrame(
                timestep: Float(timestepStr)!,
                jointPositions: jointPositions, 
                jointNames: jointNames
            )
            motionFrames.append(mf)
        }
        return motionFrames
    }

    public static func getJointPositions(motionFrames: [MotionFrame], grouppedJoints: Bool, normalized: Bool) -> ShapedArray<Float> {
        var a: Array<Array<Float>>? = nil
        if grouppedJoints {
            a = motionFrames.map {$0.grouppedJointPositions()}
        } else {
            a = motionFrames.map {$0.jointPositions}
        }
        if normalized {
            // don't sigmoid motion flag
            let mfIdx = 44
            var t = a!.makeTensor()
            let mf = t[0..., mfIdx...mfIdx]
            t = sigmoid(t)
            t[0..., mfIdx...mfIdx] = mf
            return t.array
        } else {
            return a!.makeShapedArray()
        }
    }

    public var description: String {
        return "MotionSample(timestep: \(self.motionFrames.last!.timestep), motions: \(self.motionFrames.count), annotations: \(self.annotations.count))"
    }
}

extension MotionSample {
    public static func downsampledMutlipliedMotionSamples(sampleID: Int, mmmURL: URL, annotationsURL: URL, grouppedJoints: Bool = true, normalized: Bool = true, factor: Int = 10, maxFrames: Int = 5000) -> [MotionSample] {
        let mmm_doc = MotionSample.loadMMM(fileURL: mmmURL)
        let jointNames = MotionSample.getJointNames(mmm_doc: mmm_doc)

        let motionFrames = MotionSample.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)
        let annotations = MotionSample.getAnnotations(fileURL: annotationsURL)
        let timesteps: [Float] = motionFrames.map { $0.timestep }

        var motionFramesBuckets = [[MotionFrame]](repeating: [], count: factor)
        var timestepsBuckets = [[Float]](repeating: [], count: factor)

        let nFrames = min(motionFrames.count, maxFrames)
        for idx in 0..<nFrames {
            let bucket = idx % 10
            motionFramesBuckets[bucket].append(motionFrames[idx])
            timestepsBuckets[bucket].append(timesteps[idx])
        }
        // filter out empty buckets
        let nBuckets = (nFrames>=factor) ? factor : nFrames

        return (0..<nBuckets).map {
            MotionSample(
                sampleID: sampleID, 
                motionFrames: motionFramesBuckets[$0], 
                annotations: annotations, 
                jointNames: jointNames, 
                timesteps: timestepsBuckets[$0], 
                grouppedJoints: grouppedJoints, 
                normalized: normalized
            )
        }
    }
}
