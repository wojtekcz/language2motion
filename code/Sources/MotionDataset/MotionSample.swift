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


struct MotionSample {
    public var sampleID: Int
    public var motionFrames: [MotionFrame] = []
    public var jointNames: [String] = []
    public var annotations: [String] = []
    
    init(sampleID: Int, mmmURL: URL, annotationsURL: URL) {
        self.sampleID = sampleID
        let mmm_doc = loadMMM(fileURL: mmmURL)
        self.jointNames = getJointNames(mmm_doc: mmm_doc)
        self.motionFrames = getMotionFrames(mmm_doc: mmm_doc)
        self.annotations = getAnnotations(fileURL: annotationsURL)
    }
    
    func loadMMM(fileURL: URL) -> XMLDocument {
        let mmm_text = try! String(contentsOf: fileURL, encoding: .utf8)
        return try! XMLDocument(data: mmm_text.data(using: .utf8)!, options: [])
    }
    
    func getAnnotations(fileURL: URL) -> [String] {
        let annotationsData = try! Data(contentsOf: fileURL)
        return try! JSONSerialization.jsonObject(with: annotationsData) as! [String]        
    }
    
    func getJointNames(mmm_doc: XMLDocument) -> [String] {
        let jointNode: [XMLNode] = try! mmm_doc.nodes(forXPath: "/MMM/Motion/JointOrder/Joint/@name")
        return jointNode.map {$0.stringValue!.replacingOccurrences(of: "_joint", with: "")}
    }
    
    func getMotionFrames(mmm_doc: XMLDocument) -> [MotionFrame] {
        var motionFrames: [MotionFrame] = []
        var count = 0
        for motionFrame in try! mmm_doc.nodes(forXPath: "/MMM/Motion/MotionFrames/MotionFrame") {
            count += 1
            var mf = MotionFrame(jointNames: self.jointNames)
            let tNode: [XMLNode] = try! motionFrame.nodes(forXPath:"Timestep")
            mf.timestamp = Float(tNode[0].stringValue!)!
            let jpNode: [XMLNode] = try! motionFrame.nodes(forXPath:"JointPosition")
            let jointPosition: String = jpNode[0].stringValue!            
            let comps = jointPosition.split(separator: " ")
            mf.jointPositions = comps.map {
                var xx = Float($0)
                if xx==nil { xx = 0.0 }
                return xx!
            }
            motionFrames.append(mf)
        }
        return motionFrames
    }
    
    // func getJointPositions(grouppedJoints: Bool) -> [[Float]] {
    //     if grouppedJoints {
    //         return self.motionFrames.map {$0.grouppedJointPositions()}
    //     } else {
    //         return self.motionFrames.map {$0.jointPositions}
    //     }
    // }

    func getJointPositions(grouppedJoints: Bool, normalized: Bool) -> ShapedArray<Float> {
        var a: Array<Array<Float>>? = nil
        if grouppedJoints {
            a = self.motionFrames.map {$0.grouppedJointPositions()}
        } else {
            a = self.motionFrames.map {$0.jointPositions}
        }
        if normalized {
            let t = sigmoid(tensorFromArray(arr: a!)!)
            return t.array
        } else {
            return shapedArrayFromArray(arr: a!)!
        }
    }

    func describe() -> String {
        return "MotionSample(timestamp: \(self.motionFrames.last!.timestamp), motions: \(self.motionFrames.count), annotations: \(self.annotations.count))"
    }
}
