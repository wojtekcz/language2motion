import TensorFlow

public struct TimeDistributed: Layer {
    var dense: Dense<Float>

    public init(_ wrapped: Dense<Float>) {
        self.dense = wrapped
    }

    @differentiable(wrt: (self,input))
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
        let output = dense(reshaped)
        let outputFeatures = output.shape[1]
        return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
    }
}

@differentiable
public func timeDistributed(_ input: Tensor<Float>, _ weight: Tensor<Float>) -> Tensor<Float> {
    let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
    let reshaped = input.reshaped(to: [batchSize * timeSteps, features])
    let output = matmul(reshaped, weight)
    let outputFeatures = output.shape[1]
    return output.reshaped(to: [batchSize, timeSteps, outputFeatures])
}
