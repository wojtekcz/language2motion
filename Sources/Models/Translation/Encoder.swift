//
//  Encoder.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/11/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import TextModels

/// Output of an encoder layer.
public struct EncoderLayerOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var result: Tensor<Scalar>
    @noDerivative public var attentionOutput: AttentionOutput<Scalar>?

    @differentiable
    public init(
        result: Tensor<Scalar>,
        attentionOutput: AttentionOutput<Scalar>?
    ) {
        self.result = result
        self.attentionOutput = attentionOutput
    }
}

/// Output of an encoder.
public struct EncoderOutput<Scalar: TensorFlowFloatingPoint>: Differentiable {
    public var lastLayerOutput: Tensor<Scalar>
    @noDerivative public var allLayerOutputs: [EncoderLayerOutput<Float>]

    @differentiable
    public init(
        lastLayerOutput: Tensor<Scalar>,
        allLayerOutputs: [EncoderLayerOutput<Float>]
    ) {
        self.lastLayerOutput = lastLayerOutput
        self.allLayerOutputs = allLayerOutputs
    }
}

public struct TransformerEncoderLayer2: Layer {
    public var selfAttention: MultiHeadAttention,
    feedForward: PositionwiseFeedForward,
    sublayers: [SublayerConnection]
    
    public init(size: Int, selfAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, dropoutProb: Double) {
        self.selfAttention = selfAttention
        self.feedForward = feedForward
        self.sublayers = [SublayerConnection](repeating: .init(size: size, droputProb: dropoutProb), count: 2)
    }
    
    public init(selfAttention: MultiHeadAttention, feedForward: PositionwiseFeedForward, sublayers: [SublayerConnection]) {
        self.selfAttention = selfAttention
        self.feedForward = feedForward
        self.sublayers = sublayers
    }

    @differentiable
    public func callAsFunction(_ input: TransformerInput<Float>) -> EncoderLayerOutput<Float> {
        // SR-11882
        let selfNoDerivative = withoutDerivative(at: self)
        let inputNoDerivative = withoutDerivative(at: input)
        let batchSizeNotDerivative = withoutDerivative(at: input.batchSize)
        var _attentionOutput: AttentionOutput<Float>? = nil
        var output = self.sublayers[0](.init(sequence: input.sequence, activation: {
            let attentionInput = AttentionInput(source: $0, target: $0, mask: inputNoDerivative.attentionMask, batchSize: batchSizeNotDerivative)
            let attentionOutput = selfNoDerivative.selfAttention.callAsFunction(attentionInput)
            _attentionOutput = attentionOutput
            return attentionOutput.result
        }))
        output = self.sublayers[1](.init(sequence: output, activation: {
            selfNoDerivative.feedForward.callAsFunction($0)
        }))
        return EncoderLayerOutput(result: output, attentionOutput: _attentionOutput)
    }
}

public struct Encoder: Layer {
    public var layers: [TransformerEncoderLayer2]
    public var norm: LayerNorm<Float>
    public init(layer: TransformerEncoderLayer2, layerCount: Int) {
        self.layers = [TransformerEncoderLayer2](repeating: layer, count: layerCount)
        self.norm = LayerNorm(featureCount: layerCount, axis: 2)
    }

    public init(layers: [TransformerEncoderLayer2], norm: LayerNorm<Float>) {
        self.layers = layers
        self.norm = norm
    }

    @differentiable
    public func callAsFunction(_ input: TransformerInput<Float>) -> EncoderOutput<Float> {
        var allLayerOutputs: [EncoderLayerOutput<Float>] = []
        var transformerInput = input.sequence
        
        for layerIndex in 0..<(withoutDerivative(at: layers) { $0.count }) {
            let layerOutput = layers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: input.attentionMask))
            
            let layerOutputNoDerivative = withoutDerivative(at: layerOutput) { EncoderLayerOutput(result: $0.result, attentionOutput: $0.attentionOutput) }
            allLayerOutputs.append(layerOutputNoDerivative)
            transformerInput = layerOutput.result
        }
        
        return EncoderOutput<Float>(lastLayerOutput: transformerInput, allLayerOutputs: allLayerOutputs)
    }
}
