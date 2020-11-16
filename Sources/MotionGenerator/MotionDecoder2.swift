//
//  MotionDecoder2.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import Foundation
import TensorFlow
import Datasets
import LangMotionModels


typealias DecodedSample = (motion: Tensor<Float>, log_probs: [Float], done: Tensor<Int32>)

public class MotionDecoder2 {

//    public static func greedyDecodeMotion2(sentence: LangMotionBatch.Sentence, startMotion: Tensor<Float>?,
//        transformer: LangMotionTransformer, memoryMultiplier: Float = 1.0, showAttentionProbs: Bool = false, genOpts: GenOpts)
//        -> (motion: Tensor<Float>, done: Tensor<Int32>) {
//        // print("\nEncode:")
//        // print("======")
//        let encoded = transformer.encode(input: sentence, encoderSelfAttentionTemp: genOpts.encoderSelfAttentionTemp)
//
//        if showAttentionProbs {
//            let _ = encoded.allLayerOutputs.map {tensorShow2($0.attentionOutput!.attentionProbs[0, 0])}
//        }
//
//        let memory = encoded.lastLayerOutput * memoryMultiplier
//        // print("  memory.count: \(memory.shape)")
//
//        // print("\nGenerate:")
//        // print("=========")
//
//        // start with tensor for neutral motion frame
//        let neutralMotionFrame = LangMotionBatch.neutralMotionFrame().expandingShape(at: 0)
//        var ys: Tensor<Float> = neutralMotionFrame
//        // or with supplied motion
//        if startMotion != nil {
//            ys = Tensor<Float>(concatenating: [neutralMotionFrame, startMotion!.expandingShape(at:0)], alongAxis: 1)
//        }
//        // print("ys.shape: \(ys.shape)")
//
//        var log_probs2: [Float] = []
//        var dones: [Tensor<Int32>] = []
//
//        let maxMotionLength2 = genOpts.maxMotionLength-ys.shape[1]+1
//
//        for _ in 0..<maxMotionLength2 {
//            // print("step: \(step)")
//            // print(".", terminator:"")
//            // prepare input
//            let motionPartFlag = Tensor<Int32>(repeating: 1, shape: [1, ys.shape[1]])
//            let motionPartMask = LangMotionBatch.makeSelfAttentionDecoderMask(target: motionPartFlag, pad: 0)
//            var segmentIDs = Tensor<Int32>(repeating: LangMotionBatch.MotionSegment.motion.rawValue, shape: [1, ys.shape[1]]).expandingShape(at: 2)
//            segmentIDs[0, 0, 0] = Tensor<Int32>(LangMotionBatch.MotionSegment.start.rawValue)
//            let motionPart = LangMotionBatch.MotionPart(motion: ys, decSelfAttentionMask: motionPartMask,
//                                                        motionFlag: motionPartFlag.expandingShape(at: 2), segmentIDs: segmentIDs)
//
//            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
//            // print("\(step), sourceAttentionMask.shape: \(source.sourceAttentionMask.shape)")
//            // decode motion
//            let decoded = transformer.decode(sourceMask: source.sourceAttentionMask, motionPart: motionPart, memory: memory, decoderSourceAttentionTemp: genOpts.decoderSourceAttentionTemp, decoderSelfAttentionTemp: genOpts.decoderSelfAttentionTemp)
//
//            if showAttentionProbs {
//                let _ = decoded.allLayerOutputs.map {tensorShow2($0.sourceAttentionOutput!.attentionProbs[0, 0])}
//                let _ = decoded.allLayerOutputs.map {tensorShow2($0.targetAttentionOutput!.attentionProbs[0, 0])}
//            }
//
//            // let mixtureModelInput = Tensor<Float>(concatenating: decoded.allResults, alongAxis: 2)
//            let mixtureModelInput = decoded.lastLayerOutput
//            let mixtureModelInput2 = mixtureModelInput[0...,-1].expandingShape(at: 0)//.expandingShape(at: 1)
//            let mixtureModelInput3 = transformer.preMixtureDense(mixtureModelInput2)
//            let singlePreds = transformer.mixtureModel(mixtureModelInput3)
//
//            // perform sampling
//            // let (sampledMotion, log_probs, done) = MotionDecoder.performNormalMixtureSampling(
//            //     preds: singlePreds, nb_joints: transformer.config.nbJoints, nb_mixtures: transformer.config.nbMixtures, maxMotionLength: maxMotionLength)
//            // ==================== perform sampling 100x and pick highest log_probs value
//            var samples: [DecodedSample] = []
//            for _ in 0..<genOpts.nSamples {
//                let aSample = MotionDecoder.performNormalMixtureSampling(
//                    preds: singlePreds, nb_joints: transformer.config.nbJoints, nb_mixtures: transformer.config.nbMixtures, maxMotionLength: genOpts.maxMotionLength)
//                samples.append(aSample)
//            }
//
//            // pick one with highest log_probs value
//            var bestSample: DecodedSample
//            if genOpts.bestLogProbs {
//                bestSample = samples.sorted(by: { $0.log_probs[0] > $1.log_probs[0]})[0]
//            } else {
//                bestSample = samples.sorted(by: { $0.log_probs[0] < $1.log_probs[0]})[0]
//            }
//
//            let (sampledMotion, log_probs, done) = bestSample //samples[0]
//            // ====================
//
//            // concatenate motion
//            ys = Tensor(concatenating: [ys, sampledMotion.expandingShape(at: 0)], alongAxis: 1)
//
//            // get done signal out
//            dones.append(done)
//            log_probs2.append(log_probs[0])
//        }
//        // print()
//
//        if genOpts.fixRotation {
//            let RRzIdx = MotionFrame.jpIdx(of: "RRz")
//            // print("ys.shape: \(ys.shape)")
//            // print("neutralMotionFrame.shape: \(neutralMotionFrame.shape)")
//
//            ys[0, 0..., RRzIdx] = neutralMotionFrame[0, 0, RRzIdx].broadcasted(to: [ys.shape[1]])
//        }
//
//        let dones2 = Tensor<Int32>(concatenating: dones, alongAxis: 0)
//        // print("log_probs2: \(log_probs2.reduce(0.0, +))")
//        // print(log_probs2)
//        return (motion: ys.squeezingShape(at:0)[1...], done: dones2)
//    }
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

// TODO: move to a class, move some params to init
public func greedyDecodeMotion2(textProcessor: TextProcessor, scaler: MinMaxScaler, jointNames: [String], model: LangMotionCatDistTransformer, discretizer: MotionDiscretizer, leadingFrames: SampleMotionClip?, prefix: String = "prefix", memoryMultiplier: Float = 0.0, motionsURL: URL?, showAttentionProbs: Bool = true, genOpts: GenOpts) -> Tensor<Float> {
    // returns motion: descaled groupped joints motion + motion flag tensor
    let startMotion: Tensor<Float>? = nil //getClippedMotionFrames(dataset: dataset, clipInfo: leadingFrames)
    var leadingFramesStr = "0"
    if startMotion != nil {
        leadingFramesStr = "\(startMotion!.shape[0])"
    }
    // TODO: incorporate done/stop signal
    Context.local.learningPhase = .inference
    // print("\ngreedyDecodeMotion(sentence: \"\(sentence)\")")

    let processedSentence = textProcessor.preprocess(sentence: genOpts.sentence, maxTextSequenceLength: maxTextSequenceLength)
    // processedSentence.printSentence()

    // decodedMotionFlag
//    let (decodedMotion, decodedMotionFlag) = MotionDecoder2.greedyDecodeMotion2(
//        sentence: processedSentence, startMotion: startMotion, transformer: model,
//        memoryMultiplier: memoryMultiplier, genOpts: genOpts
//    )
    let motionDecoder = MotionCatDistDecoder(discretizer: discretizer, transformer: model)
    
    let (decodedMotion, decodedMotionFlag) = motionDecoder.greedyDecodeMotion(
        sentence: processedSentence, startMotion: startMotion, maxMotionLength: 50
    )
    // print("  decodedMotion: min: \(decodedMotion.min()), max: \(decodedMotion.max())")
    let descaledMotion = scaler.inverse_transform(decodedMotion)
//    let descaledMotion = decodedMotion
    // print("  descaledMotion.shape: \(descaledMotion.shape)")
    // print("  descaledMotion: min: \(descaledMotion.min()), max: \(descaledMotion.max())")

    var imageURL: URL? = nil

    if !genOpts.saveMMM { imageURL = nil } else {
        imageURL = motionsURL!.appendingPathComponent("\(prefix).png")
    }
    // use joint groupping
    let grouppedJointsMotion = MotionSample.grouppedJoints(motion: descaledMotion, jointNames: jointNames)
    let joined = motionToImg(url: imageURL, motion: grouppedJointsMotion, motionFlag: decodedMotionFlag, padTo: genOpts.maxMotionLength, descr: "\(genOpts.sentence), LF: \(leadingFramesStr)", cmapRange: 1.0)

    if genOpts.saveMMM {
        print("Saved image: \(imageURL!.path)")
        //let jointNames = dataset.motionSamples[0].jointNames
        let mmmXMLDoc = MMMWriter.getMMMXMLDoc(jointNames: jointNames, motion: descaledMotion)
        let mmmURL = motionsURL!.appendingPathComponent("\(prefix).mmm.xml")
        // try! mmmXMLDoc.xmlData(options: XMLNode.Options.nodePrettyPrint).write(to: mmmURL)
        try! mmmXMLDoc.xmlData().write(to: mmmURL)
        print("Saved motion: \(mmmURL.path)")
    }
    return joined
}
