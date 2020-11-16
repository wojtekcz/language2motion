import Foundation
import MotionGenerator
import ModelSupport
import Datasets
import TextModels

print("MotionGenerator")

#if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
#else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
#endif

let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
var scaler = MinMaxScaler.createFromJSONURL(dataURL.appendingPathComponent("min_max_scaler.json"))!
var discretizer = MotionDiscretizer.createFromJSONURL(dataURL.appendingPathComponent("motion_discretizer.300.json"))!

let json = try! String(contentsOf: dataURL.appendingPathComponent("joint_names.json"), encoding: .utf8).data(using: .utf8)!
let jointNames: [String] = try! JSONDecoder().decode(Array<String>.self, from: json)

let runName = "run_176"
let logdir = "runs/Lang2motion/"

let mgMgr = MotionGenerationManager(scaler: scaler, jointNames: jointNames, textProcessor: textProcessor, discretizer: discretizer, logdir: logdir, runName: runName)
mgMgr.loadModel(logdir: logdir, runName: runName, modelName: "model.e3")

func generateMotion() {
    let bestLogProbs = true
    let fixRotation = true
    let saveMMM = true
    let maxMotionLength = 50
    let sentence = "A person is walking forwards."

    let opts = GenOpts(nSamples: 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMMM, encoderSelfAttentionTemp: 1.0, decoderSourceAttentionTemp: 1.0, decoderSelfAttentionTemp: 1.0, maxMotionLength: maxMotionLength, sentence: sentence)
    
    let _ = mgMgr.generateMotion(genOpts: opts, prefix: "temp_motion", model: nil)
}

generateMotion()
