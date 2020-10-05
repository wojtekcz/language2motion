import Foundation
import TensorFlow
import TextModels
import Checkpoints

public protocol ExportableLayer {
    var nameMappings: [String: String] { get }
}

extension Encoder: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "layers": "layers",
            "norm": "norm"
        ]
    }
}

extension Decoder: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "layers": "layers",
            "norm": "norm"
        ]
    }
}

extension Embedding: ExportableLayer {
    public var nameMappings: [String: String] { ["embeddings": "embeddings"] }
}

extension LayerNorm: ExportableLayer {
    public var nameMappings: [String: String] { ["offset": "offset", "scale": "scale",
                                                 // "axis": "axis", "epsilon": "epsilon"
                                                ] }
}

extension Dense: ExportableLayer {
    public var nameMappings: [String: String] { ["weight": "weight", "bias": "bias"] }
}

extension Array: ExportableLayer {
    public var nameMappings: [String: String] { ["h": "\(type(of:self))".components(separatedBy: ["<", ">"])[1] + "_h" ] }
}

extension MultiHeadAttention: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            // sourceSize: Int
            // targetSize: Int
            // headCount: Int
            // eadSize: Int
            // queryActivation: Activation<Scalar>
            // keyActivation: Activation<Scalar>
            // valueActivation: Activation<Scalar>
            // matrixResult: Bool

            "queryWeight": "queryWeight",
            "queryBias": "queryBias",
            "keyWeight": "keyWeight",
            "keyBias": "keyBias",
            "valueWeight": "valueWeight",
            "valueBias": "valueBias",
            // attentionDropout: Dropout<Scalar>
        ]
    }
}

extension TransformerEncoderLayer2: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "selfAttention": "selfAttention",
            "feedForward": "feedForward",
            "sublayers": "sublayers",
        ]
    }
}

extension PositionwiseFeedForward: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "dense1": "dense1",
            "dense2": "dense2",
            "dropout": "dropout",
        ]
    }
}

extension SublayerConnection: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "norm": "norm",
            "dropout": "dropout",
        ]
    }
}

extension TransformerDecoderLayer: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "selfAttention": "selfAttention",
            "sourceAttention": "sourceAttention",
            "feedForward": "feedForward",
            "sublayers": "sublayers",
        ]
    }
}

extension Generator: ExportableLayer {
    public var nameMappings: [String: String] {
        [
            "dense": "dense"
        ]
    }
}


public func recursivelyObtainTensors(
    _ obj: Any, scope: String? = nil, tensors: inout [String: Tensor<Float>], separator: String
) {
    
    let m = Mirror(reflecting: obj)
    let nameMappings: [String: String]
    if let exportableLayer = obj as? ExportableLayer {
        nameMappings = exportableLayer.nameMappings
    } else {
        if (obj is Int) || (obj is Bool) || (obj is Tensor<Float>) ||
           (obj is Double) || (obj is Float) || (obj is Dropout<Float>) ||
           (obj is Parameter<Float>)
        {}
        else {
            let s = "\(scope!) -> \(type(of:obj))"
            if !s.contains("Tensor") {
                // print(s)
            }
        }
        nameMappings = [:]
    }

    var repeatedLabels: [String: Int] = [:]
    func suffix(for label: String) -> String {
        if let currentSuffix = repeatedLabels[label] {
            repeatedLabels[label] = currentSuffix + 1
            return "\(currentSuffix + 1)"
        } else {
            repeatedLabels[label] = 0
            return "0"
        }
    }

    let hasSuffix = (m.children.first?.label == nil)

    var path = scope
    for child in m.children {
        let label = child.label ?? "h"

        if let remappedLabel = nameMappings[label] {
            let labelSuffix = hasSuffix ? suffix(for: remappedLabel) : ""
            let conditionalSeparator = remappedLabel == "" ? "" : separator

            path = (scope != nil ? scope! + conditionalSeparator : "") + remappedLabel + labelSuffix
            if let tensor = child.value as? Tensor<Float> {
                tensors[path!] = tensor
            }
        }
        recursivelyObtainTensors(child.value, scope: path, tensors: &tensors, separator: separator)
    }
}

public func writeCheckpoint(_ obj: Any, to location: URL, name: String) throws {
    var tensors = [String: Tensor<Float>]()
    recursivelyObtainTensors(obj, scope: "model", tensors: &tensors, separator: "/")
    // tensors.keys.sorted().map {print($0)}
    let writer = CheckpointWriter(tensors: tensors)
    try writer.write(to: location, name: name)
}
