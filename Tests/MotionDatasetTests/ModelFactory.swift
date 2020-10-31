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
            activation: relu
        )

        return LangMotionTransformer(config: config)
    }

}
