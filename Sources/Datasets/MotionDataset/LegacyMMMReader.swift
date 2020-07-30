import Foundation
import FoundationXML
import TensorFlow

public struct LegacyMMMReader {
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
            var jointPositions: [Float] = jointPositionStr.floatArray()
            jointPositions += [1.0] // Adding motion flag

            let rootPositionStr: String = (try! motionFrame.nodes(forXPath:"RootPosition"))[0].stringValue!
            let rootPosition: [Float] = rootPositionStr.floatArray()
            let rootRotationStr: String = (try! motionFrame.nodes(forXPath:"RootRotation"))[0].stringValue!
            let rootRotation: [Float] = rootRotationStr.floatArray()

            let mf = MotionFrame(
                timestep: Float(timestepStr)!,
                rootPosition: rootPosition,
                rootRotation: rootRotation,
                jointPositions: jointPositions, 
                jointNames: jointNames
            )
            motionFrames.append(mf)
        }
        return motionFrames
    }

    public static func getJointPositions(motionFrames: [MotionFrame], grouppedJoints: Bool, normalized: Bool) -> Tensor<Float> {
        var a: Array<Array<Float>>? = nil
        if grouppedJoints {
            a = motionFrames.map {$0.grouppedJointPositions()}
        } else {
            a = motionFrames.map {$0.combinedJointPositions()}
        }
        if normalized {
            // don't sigmoid motion flag
            let mfIdx = MotionFrame.cjpMotionFlagIdx
            var t = a!.makeTensor()
            let mf = t[0..., mfIdx...mfIdx]
            t = 2.0 * (sigmoid(t) - 0.5) // make range [-1.0, 1.0]
            t[0..., mfIdx...mfIdx] = mf
            return t
        } else {
            return a!.makeTensor()
        }
    }

    public static func motionSample2(sampleID: Int, mmmURL: URL, annotationsURL: URL, grouppedJoints: Bool = true, normalized: Bool = true, maxFrames: Int = 50000) -> MotionSample2 {
        let mmm_doc = LegacyMMMReader.loadMMM(fileURL: mmmURL)
        let jointNames = LegacyMMMReader.getJointNames(mmm_doc: mmm_doc)
        var motionFrames = LegacyMMMReader.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)

        let annotations = LegacyMMMReader.getAnnotations(fileURL: annotationsURL)
        var timesteps: [Float] = motionFrames.map { $0.timestep }
        if motionFrames.count > maxFrames {
            motionFrames = Array(motionFrames[0..<maxFrames])
            timesteps = Array(timesteps[0..<maxFrames])
        }
        let timestepsTensor = Tensor<Float>(shape: [timesteps.count], scalars: timesteps)
        let motion = LegacyMMMReader.getJointPositions(motionFrames: motionFrames, grouppedJoints: grouppedJoints, normalized: normalized)

        return MotionSample2(sampleID: sampleID, annotations: annotations, jointNames: jointNames, timesteps: timestepsTensor, motion: motion)
    }
}
