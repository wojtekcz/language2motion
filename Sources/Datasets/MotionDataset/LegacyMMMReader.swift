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
            let jointPositions: [Float] = jointPositionStr.floatArray()

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

    public static func getMotion(motionFrames: [MotionFrame]) -> Tensor<Float> {
        let combined: Array<Array<Float>> = motionFrames.map { $0.combinedJointPositions() }
        return combined.makeTensor()
    }

    public static func motionSampleFromMMM(sampleID: Int, mmmURL: URL, annotationsURL: URL, maxFrames: Int = 500) -> MotionSample {
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
        let motion = LegacyMMMReader.getMotion(motionFrames: motionFrames)

        return MotionSample(sampleID: sampleID, annotations: annotations, jointNames: jointNames, timesteps: timestepsTensor, motion: motion)
    }
}

extension LegacyMMMReader {
    public static func downsampledMutlipliedMotionSamples(sampleID: Int, mmmURL: URL, annotationsURL: URL, freq: Int = 10, maxFrames: Int = 500) -> [MotionSample] {
        let mmm_doc = LegacyMMMReader.loadMMM(fileURL: mmmURL)
        let jointNames = LegacyMMMReader.getJointNames(mmm_doc: mmm_doc)

        let motionFrames = LegacyMMMReader.getMotionFrames(mmm_doc: mmm_doc, jointNames: jointNames)
        let annotations = LegacyMMMReader.getAnnotations(fileURL: annotationsURL)
        let timesteps: [Float] = motionFrames.map { $0.timestep }

        // calculate factor
        let origFreq = Float(timesteps.count)/timesteps.last!
        let factor = Int(origFreq)/freq
        
        var motionFramesBuckets = [[MotionFrame]](repeating: [], count: factor)
        var timestepsBuckets = [[Float]](repeating: [], count: factor)

        for idx in 0..<motionFrames.count {
            let bucket = idx % factor
            if motionFramesBuckets[bucket].count < maxFrames {
                motionFramesBuckets[bucket].append(motionFrames[idx])
                timestepsBuckets[bucket].append(timesteps[idx])
            }
        }
        // filter out empty buckets
        let nBuckets = (motionFrames.count>=factor) ? factor : motionFrames.count

        return (0..<nBuckets).map {
            let timesteps = Tensor<Float>(shape: [timesteps.count], scalars: timesteps)
            let motion = LegacyMMMReader.getMotion(motionFrames: motionFramesBuckets[$0])
            return MotionSample(
                sampleID: sampleID, 
                annotations: annotations, 
                jointNames: jointNames, 
                timesteps: timesteps,
                motion: motion
            )
        }
    }
}
