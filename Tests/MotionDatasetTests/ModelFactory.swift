//
//  ModelFactory.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 31/10/2020.
//

import Foundation
import TensorFlow
import LangMotionModels


public class ModelFactory {

    public static func getModel(vocabSize: Int) -> LangMotionTransformer {
        let config = LangMotionTransformerConfig(
            vocabSize: vocabSize,
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
            mixtureDepth: 1500,
            activation: swish
        )

        return LangMotionTransformer(config: config)
    }

    public static func getModel2(vocabSize: Int) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 6,
            encoderDepth: 256,
            decoderDepth: 512,
            feedForwardSize: 2048,
            headCount: 16,
            dropoutProbability:  0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )

        return LangMotionCatDistTransformer(config: config)
    }


    public static func getModel3(vocabSize: Int, logdirURL: URL) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 6,
            encoderDepth: 256,
            decoderDepth: 512,
            feedForwardSize: 2048,
            headCount: 16,
            dropoutProbability:  0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )

        let model = try! LangMotionCatDistTransformer(checkpoint: logdirURL.appendingPathComponent("run_123/checkpoints"), config: config, name: "model.e1")
        return model
    }

    public static func getModel4(vocabSize: Int, logdirURL: URL) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 4,
            encoderDepth: 64,
            decoderDepth: 128,
            feedForwardSize: 256,
            headCount: 4,
            dropoutProbability: 0.1,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )

        let model = try! LangMotionCatDistTransformer(checkpoint: logdirURL.appendingPathComponent("run_125/checkpoints"), config: config, name: "model.e1")
        return model
    }

    public static func getModel5(vocabSize: Int, logdirURL: URL) -> LangMotionCatDistTransformer {
        let config = LangMotionCatDistTransformerConfig(
            vocabSize: vocabSize,
            nbJoints: 47,
            layerCount: 6,
            encoderDepth: 64,
            decoderDepth: 128,
            feedForwardSize: 512,
            headCount: 16,
            dropoutProbability: 0.0,
            sentenceMaxPositionalLength: 100,
            motionMaxPositionalLength: 500,
            discreteBins: 300,
            activation: swish
        )

        let model = try! LangMotionCatDistTransformer(checkpoint: logdirURL.appendingPathComponent("run_136/checkpoints"), config: config, name: "model.e2")
        return model
    }
}
