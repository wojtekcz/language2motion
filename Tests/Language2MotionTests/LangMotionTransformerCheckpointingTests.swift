import XCTest
import Foundation
import TensorFlow
import LangMotionModels


class LangMotionTransformerCheckpointingTests: XCTestCase {

    #if os(macOS)
        let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func testTransformerConfigWriting() throws {
        
        let vocabSize = 10000
        
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 12,
            encoderDepth: 64,
            decoderDepth: 240,
            feedForwardSize: 1536,
            headCount: 16,
            dropoutProbability: 0.0,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: .swish
        )

        let configURL = dataURL.appendingPathComponent("model_config.json")

        do {
            try config.write(to: configURL)
        }
        catch {
            print("Failed to write JSON data: \(error.localizedDescription)")
        }
    }

    func testTransformerConfigReading() throws {
        let configURL = dataURL.appendingPathComponent("model_config.json")
        let config = LangMotionCatDistTransformerConfig.createFromJSONURL(configURL)!
        print(config)
    }
    
    func getCheckpointURL() -> URL {
        return dataURL.appendingPathComponent("runs/Lang2motion").appendingPathComponent("run_181").appendingPathComponent("checkpoints")
    }
    
    func testTransformerModelWriting() throws {
        
        let vocabSize = 1000

        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 4,
            encoderDepth: 64,
            decoderDepth: 240,
            feedForwardSize: 256,
            headCount: 4,
            dropoutProbability: 0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: .swish
        )

        let model = LangMotionCatDistTransformer(config: config)
        
        let checkpointURL = getCheckpointURL()
        
        try! FileManager().createDirectory(at: checkpointURL, withIntermediateDirectories: true)
        try model.writeCheckpoint(to: checkpointURL, name: "model.t1")
        try config.write(to: checkpointURL.appendingPathComponent("model_config.json"))
    }

    func testTransformerModelReading() throws {

        let checkpointURL = getCheckpointURL()
        let config = LangMotionCatDistTransformerConfig.createFromJSONURL(checkpointURL.appendingPathComponent("model_config.json"))!
        let model = try LangMotionCatDistTransformer(checkpoint: checkpointURL, config: config, name: "model.t1")
        print(model.config.layerCount)
    }
}
