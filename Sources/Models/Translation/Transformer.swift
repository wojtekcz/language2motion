//
//  Transformer.swift
//  TranslationTransformer
//
//  Created by Andre Carrera on 2/13/20.
//  Copyright © 2020 Lambdo. All rights reserved.
//

import TensorFlow
import ModelSupport
import TextModels

public struct TransformerModel: Module {
    public var encoder: Encoder
    public var decoder: Decoder
    public var sourceEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    public var targetEmbed: Sequential<Embedding<Float>, PositionalEncoding>
    public var generator: Generator
    public init(sourceVocabSize: Int, targetVocabSize: Int, layerCount: Int = 6, modelSize: Int = 256, feedForwardSize: Int = 1024, headCount: Int = 8, dropoutProbability: Double = 0.1) {
        
        let attention = MultiHeadAttention(sourceSize: modelSize,
                                           targetSize: modelSize,
                                           headCount: headCount,
                                           headSize: modelSize/headCount,
                                           matrixResult: false)
        let feedForward = PositionwiseFeedForward(dimensionalityModel: modelSize,
                                                  innerLayerDimensionality: feedForwardSize)
        
        let positionalEncoding = PositionalEncoding(size: modelSize,
                                                    dropoutProbability: dropoutProbability)
        
        self.encoder = Encoder(layer: .init(size: modelSize, selfAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.decoder = Decoder(layer: .init(size: modelSize, selfAttention: attention, sourceAttention: attention, feedForward: feedForward, dropoutProb: dropoutProbability), layerCount: layerCount)
        self.sourceEmbed = Sequential(Embedding(vocabularySize: sourceVocabSize, embeddingSize: modelSize, embeddingsInitializer: glorotUniform()), positionalEncoding)
        self.targetEmbed = Sequential(Embedding(vocabularySize: targetVocabSize, embeddingSize: modelSize,embeddingsInitializer: glorotUniform()), positionalEncoding)
        self.generator = Generator(dimModel: modelSize, vocabSize: targetVocabSize)
    }
    
    @differentiable
    public func callAsFunction(_ input: TranslationBatch) -> Tensor<Float> {
        let encodedMemory = self.encode(input: input)
        return self.decode(input: input, memory: encodedMemory)
    }
    
    @differentiable
    public func encode(input: TranslationBatch) -> Tensor<Float> {
        let embedded = self.sourceEmbed(input.tokenIds)
        let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(input: TranslationBatch, memory: Tensor<Float>) -> Tensor<Float> {
        let embedded = self.targetEmbed(input.targetTokenIds)
        let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.mask, targetMask: input.targetMask, memory: memory)
        return self.decoder(decoderInput).lastLayerOutput
    }
    
    @differentiable
    public func generate(input: TranslationBatch) -> Tensor<Float> {
        return self.generator(self.callAsFunction(input))
    }
    @differentiable
    public func generate(input: Tensor<Float>) -> Tensor<Float> {
        self.generator(input)
    }
}

extension Tensor where Scalar == Float {
    @differentiable
    public func debugIdentity() -> Tensor<Scalar> {
        return TranslationModels.debugIdentity(self)
    }
}

public func debugIdentity(_ x: Tensor<Float>) -> Tensor<Float> {
    return x
}
@derivative(of: debugIdentity)
public func debugDerivative(_ x: Tensor<Float>) -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Float>) {
    let x_shape = x.shape
    return (x, { x_grad in
        if (x_grad.shape != x_shape) { fatalError("\(x_grad.shape) != \(x_shape)") }
        return x_grad
    })
}


public struct Generator: Layer {
    var dense: Dense<Float>
    public init(dimModel: Int, vocabSize: Int) {
        self.dense = Dense(inputSize: dimModel, outputSize: vocabSize, weightInitializer: glorotUniform())
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return logSoftmax(dense(input))
    }
}
