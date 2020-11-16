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
}
