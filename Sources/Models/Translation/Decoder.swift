//
//  Decoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import TextModels

public struct DecoderLayerOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var result: Tensor<Scalar>
    public var targetAttentionOutput: AttentionOutput<Scalar>
    public var sourceAttentionOutput: AttentionOutput<Scalar>

    @differentiable
    public init(
        result: Tensor<Scalar>,
        targetAttentionOutput: AttentionOutput<Scalar>,
        sourceAttentionOutput: AttentionOutput<Scalar>
    ) {
        self.result = result
        self.targetAttentionOutput = targetAttentionOutput
        self.sourceAttentionOutput = sourceAttentionOutput
    }
}

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
    public func callAsFunction(_ input: DecoderInput<Float>) -> DecoderLayerOutput<Float> {
        // SR-11882
        // we have to pass the input as a param in the Sublayer input because we still need to diferentiate
        // targetMask, memory, and sourceMask
        let selfNoDerivative = withoutDerivative(at: self)
        let batchSize = withoutDerivative(at: input.batchSize)
        
        let t1 = Tensor<Float>([1.0])
        var _targetAttentionOutput: AttentionOutput<Float> = AttentionOutput<Float>(result: t1, attentionProbs: t1, attentionScores: t1)
        var _sourceAttentionOutput: AttentionOutput<Float> = AttentionOutput<Float>(result: t1, attentionProbs: t1, attentionScores: t1)
        
        var output = input.sequence        
        output = self.sublayers[0].decoderForward(.init(sequence: output, decoderContext: input, activation: {
            let attentionOutput = selfNoDerivative.selfAttention(.init(source: $0,
                                                 target: $0,
                                                 mask: $1.targetMask,
                                                 batchSize: batchSize))
            _targetAttentionOutput = attentionOutput
            return attentionOutput.result
        }))
        output = self.sublayers[1].decoderForward(.init(sequence: output, decoderContext: input, activation: {
            let attentionOutput = selfNoDerivative.sourceAttention(.init(source: $0,
                                                   target: $1.memory,
                                                   mask: $1.sourceMask,
                                                   batchSize: batchSize))
            _sourceAttentionOutput = attentionOutput
            return attentionOutput.result                                      
        }))
        output = self.sublayers[2].decoderForward(.init(sequence: output, decoderContext: input, activation: {(result, _) in
            selfNoDerivative.feedForward(result)
        }))
        return DecoderLayerOutput(result: output, targetAttentionOutput: _targetAttentionOutput, sourceAttentionOutput: _sourceAttentionOutput)
    }
}

public struct DecoderOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var lastLayerOutput: Tensor<Scalar>
    public var allLayerOutputs: [DecoderLayerOutput<Float>]
    public var allResults: [Tensor<Float>]

    @differentiable
    public init(lastLayerOutput: Tensor<Scalar>, allLayerOutputs: [DecoderLayerOutput<Float>], allResults: [Tensor<Float>]) {
        self.lastLayerOutput = lastLayerOutput
        self.allLayerOutputs = allLayerOutputs
        self.allResults = allResults
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
        var allLayerOutputs: [DecoderLayerOutput<Float>] = []
        var allResults: [Tensor<Float>] = []
        var transformerInput = input.sequence
        let memoryInput = input.memory
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            let layerOutput = layers[layerIndex](DecoderInput(
                sequence: transformerInput,
                sourceMask: input.sourceMask,
                targetMask: input.targetMask,
                memory: memoryInput
            ))
            allLayerOutputs.append(layerOutput)
            allResults.append(layerOutput.result)
            transformerInput = layerOutput.result
        }
        
        return DecoderOutput<Float>(lastLayerOutput: transformerInput, allLayerOutputs: allLayerOutputs, allResults: allResults)
    }
}
