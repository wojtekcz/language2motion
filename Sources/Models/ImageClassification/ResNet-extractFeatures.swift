import TensorFlow

extension ResNet {
    @differentiable
    public func extractFeatures(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputLayer = maxPool(relu(initialLayer(input)))
        let blocksReduced = residualBlocks.differentiableReduce(inputLayer) { last, layer in
            layer(last)
        }
        return blocksReduced.sequenced(through: avgPool, flatten)
    }
}
