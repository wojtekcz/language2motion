import TensorFlow

public struct MotionGaussianMixtureModel: Layer {

    @noDerivative public var inputSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int
    @noDerivative public var outputSize: Int

    public var linearMixtureMeans: Dense<Float>
    public var linearMixtureVars: Dense<Float>
    public var linearMixtureWeights: Dense<Float>
    public var linearStop: Dense<Float>

    public init(inputSize: Int, nbJoints: Int, nbMixtures: Int) {
        self.inputSize = inputSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.outputSize = MotionGaussianMixtureModel.getOutputSize(nbJoints: nbJoints, nbMixtures: nbMixtures)

        // FC layers for learning gaussian mixture distributions params
        // input_size  // hidden_size*n_layers, FIXME: Plappert concatenates output of all rnn layers
        linearMixtureMeans = Dense<Float>(inputSize: inputSize, outputSize: nbJoints * nbMixtures)
        linearMixtureVars = Dense<Float>(inputSize: inputSize, outputSize: nbJoints * nbMixtures)
        linearMixtureWeights = Dense<Float>(inputSize: inputSize, outputSize: nbMixtures)

        // and stop bit
        linearStop = Dense<Float>(inputSize: inputSize, outputSize: 1)
    }

    @differentiable
    func forwardStep(_ x: Tensor<Float>) -> Tensor<Float> {
        // bs x input_size
        // Processing gaussian mixture params:
        let mixtureMeans = linearMixtureMeans(x)
        let mixtureVars = softplus(linearMixtureVars(x))
        let mixtureWeights = softmax(linearMixtureWeights(x), alongAxis: 1)
        // stop
        let stop = sigmoid(linearStop(x))
        // merge
        let mixtureMerged = Tensor(concatenating: [mixtureMeans, mixtureVars, mixtureWeights, stop])
        return mixtureMerged // bs x output_size
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x = input
        let bs = x.shape[0]
        let max_target_length = x.shape[1]
        var all_decoder_outputs = Tensor<Float>(zeros: [bs, max_target_length, self.outputSize])
        // add neutral position vector?
        // Run through mixture_model one time step at a time
        for t in 0..<max_target_length-1 {
            let decoder_output: Tensor<Float> = x[0..., t]

            all_decoder_outputs[0..., t] = self.forwardStep(decoder_output)
        }
        return all_decoder_outputs
    }

    static func getOutputSize(nbJoints: Int, nbMixtures: Int) -> Int {
        return 2 * nbMixtures * nbJoints + nbMixtures + 1
    }
}
