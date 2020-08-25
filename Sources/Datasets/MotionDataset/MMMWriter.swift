import FoundationXML
import TensorFlow


public class MMMWriter {
    // TODO: work with MotionSample
    
    public static func getSensorModelPoseElem(frequency: Float, RootPositionZ: Int = 520, rootRotation: Tensor<Float>) -> XMLElement {
        // rootRotation shape [numFrames, 3]
        let sensorModelPoseElem = XMLElement(name: "Sensor")

        sensorModelPoseElem.addAttribute(XMLNode.attribute(withName: "type", stringValue: "ModelPose") as! XMLNode)
        sensorModelPoseElem.addAttribute(XMLNode.attribute(withName: "version", stringValue: "1.0") as! XMLNode)

        let modelPoseConfigurationElem = XMLElement(name: "Configuration")

        let modelPoseDataElem = XMLElement(name: "Data")

        for i in 0..<rootRotation.shape[0] {
            let timestep = Float(i)/frequency
            let measurementElem = XMLElement(name: "Measurement")
            measurementElem.addAttribute(XMLNode.attribute(withName: "timestep", stringValue: "\(timestep)") as! XMLNode)

            let rootPositionElem = XMLElement(name: "RootPosition", stringValue: "0 0 \(RootPositionZ)")
            measurementElem.addChild(rootPositionElem)

            let rootRotationArr: [String] = rootRotation[i].scalars.map { "\($0)"}    
            let rootRotationElem = XMLElement(name: "RootRotation", stringValue: rootRotationArr.joined(separator: " "))
            measurementElem.addChild(rootRotationElem)
            modelPoseDataElem.addChild(measurementElem)
        }

        sensorModelPoseElem.addChild(modelPoseConfigurationElem)
        sensorModelPoseElem.addChild(modelPoseDataElem)

        return sensorModelPoseElem
    }
    
    public static func getSensorKinematicElem(frequency: Float, jointNames: [String], jointPosition: Tensor<Float>) -> XMLElement {
        //rootRotation shape [numFrames, 44]
        let sensorKinematicElem = XMLElement(name: "Sensor")

        sensorKinematicElem.addAttribute(XMLNode.attribute(withName: "type", stringValue: "Kinematic") as! XMLNode)
        sensorKinematicElem.addAttribute(XMLNode.attribute(withName: "version", stringValue: "1.0") as! XMLNode)

        let kinematicConfigurationElem = XMLElement(name: "Configuration")


        for jointName in jointNames {
            let jointElem = XMLElement(name: "Joint")
            jointElem.addAttribute(XMLNode.attribute(withName: "name", stringValue: "\(jointName)_joint") as! XMLNode)
            kinematicConfigurationElem.addChild(jointElem)
        }

        sensorKinematicElem.addChild(kinematicConfigurationElem)

        let kinematicDataElem = XMLElement(name: "Data")

        for i in 0..<jointPosition.shape[0] {
            let timestep = Float(i)/frequency
            let measurementElem = XMLElement(name: "Measurement")
            measurementElem.addAttribute(XMLNode.attribute(withName: "timestep", stringValue: "\(timestep)") as! XMLNode)

            let jointPositionArr: [String] = jointPosition[i].scalars.map { "\($0)"}

            let jointPositionElem = XMLElement(name: "JointPosition", stringValue: jointPositionArr.joined(separator: " "))
            measurementElem.addChild(jointPositionElem)
            kinematicDataElem.addChild(measurementElem)
        }

        sensorKinematicElem.addChild(kinematicDataElem)

        return sensorKinematicElem
    }
    
    public static func getMMMXMLDoc(jointNames: [String], motion: Tensor<Float>) -> XMLDocument {
        let xmlDoc = XMLDocument()

        let mmmElem = XMLElement(name: "MMM")
        let verAttr: XMLNode = XMLNode.attribute(withName: "version", stringValue: "2.0") as! XMLNode
        mmmElem.addAttribute(verAttr)
        xmlDoc.addChild(mmmElem)

        let motionElem = XMLElement(name: "Motion")
        let modelElem = XMLElement(name: "Model")
        let sensorsElem = XMLElement(name: "Sensors")

        motionElem.addAttribute(XMLNode.attribute(withName: "name", stringValue: "export") as! XMLNode)
        motionElem.addAttribute(XMLNode.attribute(withName: "synchronized", stringValue: "true") as! XMLNode)

        modelElem.addAttribute(XMLNode.attribute(withName: "path", stringValue: "mmm.xml") as! XMLNode)

        let sensorModelPoseElem = getSensorModelPoseElem(frequency: 10, RootPositionZ: 520, rootRotation: motion[0..., 44..<47])
        let sensorKinematicElem = getSensorKinematicElem(frequency: 10, jointNames: jointNames, jointPosition: motion[0..., 0..<44])

        mmmElem.addChild(motionElem)
        motionElem.addChild(modelElem)
        motionElem.addChild(sensorsElem)
        sensorsElem.addChild(sensorModelPoseElem)
        sensorsElem.addChild(sensorKinematicElem)

        return xmlDoc
    }
}
