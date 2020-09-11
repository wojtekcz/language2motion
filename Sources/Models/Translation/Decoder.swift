//
//  Decoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import TextModels

public struct TransformerDecoderLayer: Layer {
    public var selfAttention: MultiHeadAttention,
    sourceAttention: MultiHeadAttention,
    feedForward: PositionwiseFeedForward,
    sublayers: [SublayerConnection]
    
    public init(size: Int, selfAttention: MultiHeadAttention, sourceAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, dropoutProb: Double) {
        self.selfAttention = selfAttention
        self.sourceAttention = sourceAttention
        self.feedForward = feedForward
        self.sublayers = [SublayerConnection](repeating: .init(size: size, droputProb: dropoutProb), count: 3)
    }

    public init(selfAttention: MultiHeadAttention, sourceAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, sublayers: [SublayerConnection]) {
        self.selfAttention = selfAttention
        self.sourceAttention = sourceAttention
        self.feedForward = feedForward
        self.sublayers = sublayers
    }

    @differentiable
    public func callAsFunction(_ input: DecoderInput<Float>) -> Tensor<Float> {
        // SR-11882
        // we have to pass the input as a param in the Sublayer input because we still need to diferentiate
        // targetMask, memory, and sourceMask
        let selfNoDerivative = withoutDerivative(at: self)
        let batchSize = withoutDerivative(at: input.batchSize)
        
        var output = input.sequence
        
        
        output = self.sublayers[0].decoderForward(.init(sequence: output, decoderContext: input, activation: {
            selfNoDerivative.selfAttention(.init(source: $0,
                                                 target: $0,
                                                 mask: $1.targetMask,
                                                 batchSize: batchSize))
        }))
        output = self.sublayers[1].decoderForward(.init(sequence: output, decoderContext: input, activation: {
            selfNoDerivative.sourceAttention(.init(source: $0,
                                                   target: $1.memory,
                                                   mask: $1.sourceMask,
                                                   batchSize: batchSize))
        }))
        output = self.sublayers[2].decoderForward(.init(sequence: output, decoderContext: input, activation: {(result, _) in
            selfNoDerivative.feedForward(result)
        }))
        return output
    }
}

public struct DecoderOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var lastLayerOutput: Tensor<Scalar>
    public var allOutputs: [Tensor<Scalar>]

    @differentiable
    public init(lastLayerOutput: Tensor<Scalar>, allOutputs: [Tensor<Scalar>]) {
        self.lastLayerOutput = lastLayerOutput
        self.allOutputs = allOutputs
    }
}

public struct Decoder: Layer {
    public var layers: [TransformerDecoderLayer]
    public var norm: LayerNorm<Float>
    public init(layer: TransformerDecoderLayer, layerCount: Int) {
        self.layers = [TransformerDecoderLayer](repeating: layer, count: layerCount)
        self.norm = LayerNorm(featureCount: layerCount, axis: 2)
    }
    
    public init(layers: [TransformerDecoderLayer], norm: LayerNorm<Float>) {
        self.layers = layers
        self.norm = norm
    }

    @differentiable
    public func callAsFunction(_ input: DecoderInput<Float>) -> DecoderOutput<Float> {
        var allOutputs: [Tensor<Float>] = []
        var transformerInput = input.sequence
        let memoryInput = input.memory
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            let layerOutput = layers[layerIndex](DecoderInput(
                sequence: transformerInput,
                sourceMask: input.sourceMask,
                targetMask: input.targetMask,
                memory: memoryInput
            ))
            allOutputs.append(layerOutput)
            transformerInput = layerOutput
        }
        
        return DecoderOutput<Float>(lastLayerOutput: transformerInput, allOutputs: allOutputs)
    }
}
