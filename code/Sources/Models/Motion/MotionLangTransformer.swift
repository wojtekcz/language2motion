import TensorFlow
import Datasets
import TranslationModels

// Transformer with MotionLangBatch

extension TransformerModel {
    @differentiable
    public func callAsFunction(_ input: MotionLangBatch) -> Tensor<Float> {
        let encodedMemory = self.encode(input: input)
        return self.decode(input: input, memory: encodedMemory)
    }
    
    @differentiable
    public func encode(input: MotionLangBatch) -> Tensor<Float> {
        // let embedded = self.sourceEmbed(input.tokenIds)
        // let encoderInput = TransformerInput(sequence: embedded, attentionMask: input.mask)
        let embedded = self.sourceEmbed(input.targetTokenIds)
        let encoderInput: TransformerInput = TransformerInput(sequence: embedded, attentionMask: input.targetMask)
        return self.encoder(encoderInput)
    }
    
    @differentiable
    public func decode(input: MotionLangBatch, memory: Tensor<Float>) -> Tensor<Float> {
        let embedded = self.targetEmbed(input.targetTokenIds)
        let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.targetMask, targetMask: input.targetMask, memory: memory)
        // let decoderInput = DecoderInput(sequence: embedded, sourceMask: input.mask, targetMask: input.targetMask, memory: memory)
        return self.decoder(decoderInput)
    }
    
    @differentiable
    public func generate(input: MotionLangBatch) -> Tensor<Float> {
        return self.generator(self.callAsFunction(input))
    }
}
