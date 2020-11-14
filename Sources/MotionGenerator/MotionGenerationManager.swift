//
//  MotionGenerationManager.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import Foundation
import TensorFlow
import ModelSupport
import Datasets
import TextModels
import LangMotionModels
import PythonKit

let plt = Python.import("matplotlib.pyplot")

let maxTextSequenceLength =  40

public struct GenOpts {
    
    public let nSamples: Int
    public let bestLogProbs: Bool
    public let fixRotation: Bool
    public let saveMMM: Bool
    
    public let encoderSelfAttentionTemp: Float
    public let decoderSourceAttentionTemp: Float
    public let decoderSelfAttentionTemp: Float
    
    public let maxMotionLength: Int

    public let sentence: String
    
    public init(nSamples: Int, bestLogProbs: Bool, fixRotation: Bool, saveMMM: Bool, encoderSelfAttentionTemp: Float, decoderSourceAttentionTemp: Float, decoderSelfAttentionTemp: Float, maxMotionLength: Int, sentence: String) {
        self.nSamples = nSamples
        self.bestLogProbs = bestLogProbs
        self.fixRotation = fixRotation
        self.saveMMM = saveMMM
        self.encoderSelfAttentionTemp = encoderSelfAttentionTemp
        self.decoderSourceAttentionTemp = decoderSourceAttentionTemp
        self.decoderSelfAttentionTemp = decoderSelfAttentionTemp
        self.maxMotionLength = maxMotionLength
        self.sentence = sentence
    }
}

public class MotionGenerationManager {
    var dataset: Lang2Motion?
    var model: LangMotionCatDistTransformer?
    var vocabulary: Vocabulary?
    var textProcessor: TextProcessor?
    var discretizer: MotionDiscretizer?

    var epoch = 1
    var motionsURL: URL?
    var genNum = 1

    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")

    public init() {
    }

    public init(dataset: Lang2Motion, textProcessor: TextProcessor, discretizer: MotionDiscretizer, logdir: String, runName: String) {
        self.dataset = dataset
        self.textProcessor = textProcessor
        self.discretizer = discretizer
        
        let logdirURL = dataURL.appendingPathComponent(logdir, isDirectory: true)
        let runURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
        motionsURL = runURL.appendingPathComponent("generated_motions_app", isDirectory: true)
        try! FileManager().createDirectory(at: motionsURL!, withIntermediateDirectories: true)
    }

    public func loadDataset(datasetSize: DatasetSize, maxSamples: Int? = nil, maxMotionLength: Int = 75) {
        let device = Device.defaultTFEager
        let batchSize = 2
        let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")
        
        /// instantiate text processor
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary!, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        textProcessor = TextProcessor(vocabulary: vocabulary!, tokenizer: tokenizer)
        discretizer = MotionDiscretizer(n_bins: 300)
        
        print("\nLoading dataset...")

        dataset = try! Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: batchSize,
            minMotionLength: 10,
            maxMotionLength: maxMotionLength,
            maxSamples: maxSamples,
            discretizer: &discretizer!,
            trainTestSplit: 1.0,
            device: device
        ) { (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = self.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: self.discretizer!)
            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source,label: target)
            return singleBatch
        }

        print("Dataset acquired.")
    }
    
    public func loadModel(logdir: String, runName: String, modelName: String) {
        let logdirURL = dataURL.appendingPathComponent(logdir, isDirectory: true)
        model = getModel5(vocabSize: vocabulary!.count, logdirURL: logdirURL, runName: runName, modelName: modelName)
        print("Loaded.")
        
    }

    public func getModel4(vocabSize: Int, logdirURL: URL) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 6,
            encoderDepth: 256,
            decoderDepth: 512,
            feedForwardSize: 2048,
            headCount: 16,
            dropoutProbability: 0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )
        
        let runName = "run_160_maxSamples"
        let runURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
        let checkpointURL = runURL.appendingPathComponent("checkpoints", isDirectory: true)
        motionsURL = runURL.appendingPathComponent("generated_motions_app", isDirectory: true)
        try! FileManager().createDirectory(at: motionsURL!, withIntermediateDirectories: true)

        let model = try! LangMotionCatDistTransformer(checkpoint: checkpointURL, config: config, name: "model.e44")
        return model
    }

    public func getModel5(vocabSize: Int, logdirURL: URL, runName: String, modelName: String) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 12,
            encoderDepth: 64,
            decoderDepth: 240,
            feedForwardSize: 1536,
            headCount: 16,
            dropoutProbability: 0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )
        
        //let runName = "run_176"
        let runURL = logdirURL.appendingPathComponent(runName, isDirectory: true)
        let checkpointURL = runURL.appendingPathComponent("checkpoints", isDirectory: true)
        motionsURL = runURL.appendingPathComponent("generated_motions_app", isDirectory: true)
        try! FileManager().createDirectory(at: motionsURL!, withIntermediateDirectories: true)

        let model = try! LangMotionCatDistTransformer(checkpoint: checkpointURL, config: config, name: modelName)
        return model
    }

    public func generateMotion(genOpts: GenOpts, prefix: String = "temp_motion", model: LangMotionCatDistTransformer?) -> Tensor<Float> {
//        print("generateMotion()")
        let lf: SampleMotionClip? = nil

//        let prefix = "epoch_\(epoch)_motion_\(genNum)"
//        let prefix = "temp_motion"
        
        let usedModel = model ?? self.model!
        
        let joined = greedyDecodeMotion2(textProcessor: textProcessor!, dataset: dataset!, model: usedModel, discretizer: discretizer!, leadingFrames: lf,
            prefix: prefix, memoryMultiplier: 1.0, motionsURL: motionsURL!, showAttentionProbs: false, genOpts: genOpts)
        
        genNum += 1
        return joined
    }
}


func tensorShow2(_ tensor: Tensor<Float>) {
    plt.imshow(tensor.makeNumpyArray(), cmap: "Spectral")
    plt.show()
}

func saveMotionToMMM(dataset: Lang2Motion, motion: Tensor<Float>, mmmURL: URL) {
    let descaledMotion = dataset.scaler.inverse_transform(motion)
//    let descaledMotion = motion
    let jointNames = dataset.motionSamples[0].jointNames
    let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
    try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
//    print("Saved motion: \(mmmURL.path)")
}

//func showMotionSample(dataset: Lang2Motion, _ motionSample: MotionSample) {
//    let motion = motionSample.motion
//    let descaledMotion = dataset.scaler.inverse_transform(motion)
//    let sentence = "sample_id=\(motionSample.sampleID), ann=\(motionSample.annotations[0])"
//
//    print("motion: min: \(motion.min()), max: \(motion.max())")
//    print("descaledMotion.shape: \(descaledMotion.shape)")
//    print("descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")
//
//    // use joint groupping
//    let jointNames = dataset.motionSamples[0].jointNames
//    let grouppedJointsMotion = MotionSample.grouppedJoints(motion: descaledMotion, jointNames: dataset.motionSamples[0].jointNames)
//    motionToImg(url: nil, motion: grouppedJointsMotion, motionFlag: nil, padTo: maxMotionLength, descr: sentence, cmapRange: 1.0)
//}

//func showMotion(dataset: Lang2Motion, motion: Tensor<Float>) {
//    let descaledMotion = dataset.scaler.inverse_transform(motion)
//    let grouppedJointsMotion = MotionSample.grouppedJoints(motion: descaledMotion, jointNames: dataset.motionSamples[0].jointNames)
//    motionToImg(url: nil, motion: grouppedJointsMotion, motionFlag: nil, padTo: maxMotionLength, descr: "", cmapRange: 1.0)
//}
