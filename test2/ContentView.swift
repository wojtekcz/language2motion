//
//  ContentView.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import SwiftUI
import TensorFlow

import TextModels
import TranslationModels
import Foundation
import ModelSupport
import Datasets
import SummaryWriter
import LangMotionModels
import Checkpoints
import PythonKit

let maxTextSequenceLength =  40
let maxMotionLength = 150


let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")

func tensorShow2(_ tensor: Tensor<Float>) {
    plt.imshow(tensor.makeNumpyArray(), cmap: "Spectral")
    plt.show()
}

typealias DecodedSample = (motion: Tensor<Float>, log_probs: [Float], done: Tensor<Int32>)

public class MotionDecoder2 {

// extension MotionDecoder2 {
    public static func greedyDecodeMotion2(
        sentence: LangMotionBatch.Sentence,
        startMotion: Tensor<Float>?,
        transformer: LangMotionTransformer,
        maxMotionLength: Int,
        memoryMultiplier: Float = 1.0,
        showAttentionProbs: Bool = false,
        bestLogProbs: Bool = true
    ) -> (motion: Tensor<Float>, done: Tensor<Int32>) {
        print("\nEncode:")
        print("======")
        let encoded = transformer.encode(input: sentence)
        
        if showAttentionProbs {
            encoded.allLayerOutputs.map {tensorShow2($0.attentionOutput!.attentionProbs[0, 0])}
        }
        
        let memory = encoded.lastLayerOutput * memoryMultiplier
        print("  memory.count: \(memory.shape)")

        print("\nGenerate:")
        print("=========")

        // start with tensor for neutral motion frame
        let neutralMotionFrame = LangMotionBatch.neutralMotionFrame().expandingShape(at: 0)
        var ys: Tensor<Float> = neutralMotionFrame
        // or with supplied motion
        if startMotion != nil {
            ys = Tensor<Float>(concatenating: [neutralMotionFrame, startMotion!.expandingShape(at:0)], alongAxis: 1)
        }
        print("ys.shape: \(ys.shape)")

        var log_probs2: [Float] = []
        var dones: [Tensor<Int32>] = []

        let maxMotionLength2 = maxMotionLength-ys.shape[1]+1

        for step in 0..<maxMotionLength2 {
            // print("step: \(step)")
            print(".", terminator:"")
            // prepare input
            let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
            let motionPartMask = LangMotionBatch.makeSelfAttentionDecoderMask(target: motionPartFlag, pad: 0)
            var segmentIDs = Tensor<Int32>(repeating: LangMotionBatch.MotionSegment.motion.rawValue, shape: [1, ys.shape[1]]).expandingShape(at: 2)
            segmentIDs[0, 0, 0] = Tensor<Int32>(LangMotionBatch.MotionSegment.start.rawValue)
            let motionPart = LangMotionBatch.MotionPart(motion: ys, decSelfAttentionMask: motionPartMask,
                                                        motionFlag: motionPartFlag.expandingShape(at: 2), segmentIDs: segmentIDs)

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            // print("\(step), sourceAttentionMask.shape: \(source.sourceAttentionMask.shape)")
            // decode motion
            let decoded = transformer.decode(sourceMask: source.sourceAttentionMask, motionPart: motionPart, memory: memory)

            if showAttentionProbs {
                decoded.allLayerOutputs.map {tensorShow2($0.sourceAttentionOutput!.attentionProbs[0, 0])}
                decoded.allLayerOutputs.map {tensorShow2($0.targetAttentionOutput!.attentionProbs[0, 0])}
            }

            // let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
            let mixtureModelInput = decoded.lastLayerOutput
            let mixtureModelInput2 = mixtureModelInput[0...,-1].expandingShape(at: 0)
            let singlePreds = transformer.mixtureModel(mixtureModelInput2)
            
            // perform sampling
//             let (sampledMotion, log_probs, done) = MotionDecoder.performNormalMixtureSampling(
//                 preds: singlePreds, nb_joints: transformer.config.nbJoints, nb_mixtures: transformer.config.nbMixtures, maxMotionLength: maxMotionLength)
            // ==================== perform sampling 100x and pick highest log_probs value
            var samples: [DecodedSample] = []
            for x in 0..<100 {
                let aSample = MotionDecoder.performNormalMixtureSampling(
                    preds: singlePreds, nb_joints: transformer.config.nbJoints, nb_mixtures: transformer.config.nbMixtures, maxMotionLength: maxMotionLength)
                samples.append(aSample)
            }

            // pick one with highest log_probs value
            var bestSample: DecodedSample
            if bestLogProbs {
                bestSample = samples.sorted(by: { $0.log_probs[0] > $1.log_probs[0]})[0]
            } else {
                bestSample = samples.sorted(by: { $0.log_probs[0] < $1.log_probs[0]})[0]
            }

            let (sampledMotion, log_probs, done) = bestSample //samples[0]
            // ====================
            
            // concatenate motion
            ys = Tensor(concatenating: [ys, sampledMotion.expandingShape(at: 0)], alongAxis: 1)
            
            // get done signal out
            dones.append(done)
            log_probs2.append(log_probs[0])
        }
        print()
        let dones2 = Tensor<Int32>(concatenating: dones, alongAxis: 0)
        print("log_probs2: \(log_probs2.reduce(0.0, +))")
        print(log_probs2)
        return (motion: ys.squeezingShape(at:0)[1...], done: dones2)
    }
}

public struct SampleMotionClip {
    var sampleID: Int
    var start: Int = 0
    var length: Int = 1
}

public func getClippedMotionFrames(dataset: Lang2Motion, clipInfo: SampleMotionClip?) -> Tensor<Float>? {
    if clipInfo != nil {

    let ms: MotionSample = dataset.motionSamples.filter { $0.sampleID == clipInfo!.sampleID } [0]
    let clippedMotionFrames = ms.motion[clipInfo!.start..<clipInfo!.start+clipInfo!.length]
    return clippedMotionFrames
    } else {
        return nil
    }
}

public func greedyDecodeMotion2(textProcessor: TextProcessor, dataset: Lang2Motion, model: LangMotionTransformer, sentence: String, leadingFrames: SampleMotionClip?, prefix: String = "prefix",
                                saveMotion: Bool = true, memoryMultiplier: Float = 0.0, motionsURL: URL?, maxMotionLength: Int, showAttentionProbs: Bool = true, bestLogProbs: Bool = true) {
    let startMotion: Tensor<Float>? = getClippedMotionFrames(dataset: dataset, clipInfo: leadingFrames)
    var leadingFramesStr = "0"
    if startMotion != nil {
        leadingFramesStr = "\(startMotion!.shape[0])"
    }
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let processedSentence = textProcessor.preprocess(sentence: sentence, maxTextSequenceLength: maxTextSequenceLength)
    processedSentence.printSentence()

    let (decodedMotion, decodedMotionFlag) = MotionDecoder2.greedyDecodeMotion2(
        sentence: processedSentence,
        startMotion: startMotion,
        transformer: model,
        maxMotionLength: maxMotionLength,
        memoryMultiplier: memoryMultiplier,
        bestLogProbs: bestLogProbs
    )
    print("  decodedMotion: min: \(decodedMotion.min()), max: \(decodedMotion.max())")
    let descaledMotion = dataset.scaler.inverse_transform(decodedMotion)
    print("  descaledMotion.shape: \(descaledMotion.shape)")
    print("  descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")

    var imageURL: URL? = nil
    
    if !saveMotion { imageURL = nil } else {
        imageURL = motionsURL!.appendingPathComponent("\(prefix).png")
    }
    // use joint groupping
    let grouppedJointsMotion = MotionSample.grouppedJoints(motion: descaledMotion, jointNames: dataset.motionSamples[0].jointNames)
    motionToImg(url: imageURL, motion: grouppedJointsMotion, motionFlag: decodedMotionFlag, padTo: maxMotionLength, descr: "\(sentence), LF: \(leadingFramesStr)", cmapRange: 1.0)

    if saveMotion {
        print("Saved image: \(imageURL!.path)")
        let jointNames = dataset.motionSamples[0].jointNames
        let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
        let mmmURL = motionsURL!.appendingPathComponent("\(prefix).mmm.xml")
        try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
        print("Saved motion: \(mmmURL.path)")
    }
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

func saveMotionToMMM(dataset: Lang2Motion, motion: Tensor<Float>, mmmURL: URL) {
    let descaledMotion = dataset.scaler.inverse_transform(motion)
    let jointNames = dataset.motionSamples[0].jointNames
    let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
    try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
    print("Saved motion: \(mmmURL.path)")
}

var dataset: Lang2Motion?
var model: LangMotionTransformer?
var vocabulary: Vocabulary?
var textProcessor: TextProcessor?

var epoch = 1
var motionsURL: URL?

struct ContentView: View {
    @State var nSamples: String
    
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
        
    var body: some View {
        VStack {
            Text("Hello, World!")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Button(action: loadDataset) {
                Text("Load dataset")
            }
            Button(action: loadModel) {
                Text("Load model")
            }
            Button(action: generateMotion) {
                Text("Generate motion")
            }
            HStack {
                Text("nSamples")
                TextField("samples", text: $nSamples)
            }
        }
    }
    

    func loadDataset() {
        let device = Device.defaultTFEager
                
        let datasetSize: DatasetSize = .full
        let batchSize = 150
        
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
            let sentence = textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: maxTextSequenceLength)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength)
            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source,label: target)
            return singleBatch
        }

        print("Dataset acquired.")
    }

    func loadModel() {
        /// Load model checkpoint
        let runName = "run_74"
        epoch = 10

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
    
    func generateMotion() {
        print("Ala ma kota")
        print("nSamples: \(nSamples)")
        let t1 = Tensor<Float>([1.0, 2.0, 3.0])
        print(t1 .* t1)
        
        var genNum = 1

        var s: String = ""
        var lf: SampleMotionClip?

        s = "A person is walking forwards."
        lf = nil

        greedyDecodeMotion2(textProcessor: textProcessor!, dataset: dataset!, model: model!, sentence: s, leadingFrames: lf,
            prefix: "epoch_\(epoch)_motion_\(genNum)",
            saveMotion: true, memoryMultiplier: 1.0, motionsURL: motionsURL!,
            maxMotionLength: 100, showAttentionProbs: false, bestLogProbs: true
        )
        genNum += 1
        
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(nSamples: "10")
    }
}
