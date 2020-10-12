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


public class MotionGenerationManager {
    var dataset: Lang2Motion?
    var model: LangMotionTransformer?
    var vocabulary: Vocabulary?
    var textProcessor: TextProcessor?

    var epoch = 1
    var motionsURL: URL?
    var genNum = 1

    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")

    func loadDataset() {
        let device = Device.defaultTFEager
                
        let datasetSize: DatasetSize = .full
        let batchSize = 150
        let maxMotionLength = 150

        let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")
        
        /// instantiate text processor
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary!, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        textProcessor = TextProcessor(vocabulary: vocabulary!, tokenizer: tokenizer)
        
        print("\nLoading dataset...")

        dataset = try! Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: batchSize,
            minMotionLength: 20,
            maxMotionLength: 150,
            trainTestSplit: 1.0,
            device: device
        ) { (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = self.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength)
            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source,label: target)
            return singleBatch
        }

        print("Dataset acquired.")
    }

    func loadModel() {
        /// Load model checkpoint
        let runName = "full_7x/run_75"
        epoch = 100

        // let encoderSelfAttentionTemp = 1000.0
        // let decoderSourceAttentionTemp = 1000.0
        // let decoderSelfAttentionTemp = 1000.0

        let encoderSelfAttentionTemp = 1.0
        let decoderSourceAttentionTemp = 1.0
        let decoderSelfAttentionTemp = 100000.0
        
        let runURL = dataURL.appendingPathComponent("runs/Lang2motion/\(runName)", isDirectory: true)
        let checkpointURL = runURL.appendingPathComponent("checkpoints", isDirectory: true)
        motionsURL = runURL.appendingPathComponent("generated_motions_app", isDirectory: true)
        try! FileManager().createDirectory(at: motionsURL!, withIntermediateDirectories: true)

        let config = LangMotionTransformerConfig(
            vocabSize: vocabulary!.count,
            nbJoints: 47,
            nbMixtures: 20,
            layerCount: 6,
            encoderDepth: 256,
            decoderDepth: 512,
            feedForwardSize: 2048,
            headCount: 16,
            dropoutProbability:  0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            encoderSelfAttentionTemp: encoderSelfAttentionTemp,
            decoderSourceAttentionTemp: decoderSourceAttentionTemp,
            decoderSelfAttentionTemp: decoderSelfAttentionTemp
        )

        model = try! LangMotionTransformer(checkpoint: checkpointURL, config: config, name: "model.e\(epoch)")
        print("Loaded.")
        
    }
    
    func generateMotion(genOpts: GenOpts) -> Tensor<Float> {
//        print("generateMotion()")
        let lf: SampleMotionClip? = nil

        let joined = greedyDecodeMotion2(textProcessor: textProcessor!, dataset: dataset!, model: model!, sentence: genOpts.sentence, leadingFrames: lf,
            prefix: "epoch_\(epoch)_motion_\(genNum)",
            saveMotion: genOpts.saveMMM, memoryMultiplier: 1.0, motionsURL: motionsURL!,
            maxMotionLength: genOpts.maxMotionLength, showAttentionProbs: false, bestLogProbs: genOpts.bestLogProbs, nSamples: genOpts.nSamples
        )
        
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
    let jointNames = dataset.motionSamples[0].jointNames
    let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
    try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
    print("Saved motion: \(mmmURL.path)")
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
