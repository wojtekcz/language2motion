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
    // def forward_step(self, x):  # bs x input_size
    //     # Note: we run this one step at a time

    //     # Processing gaussian mixture params:
    //     mixture_means = self.linear_mixture_means(x)
    //     mixture_vars = self.softplus(self.linear_mixture_vars(x))
    //     mixture_weights = self.softmax(self.linear_mixture_weights(x))
    //     # stop
    //     stop = self.sigmoid(self.linear_stop(x))
    //     # merge
    //     mixture_merged = torch.cat((mixture_means, mixture_vars, mixture_weights, stop), 1)

    //     return mixture_merged  # bs x output_size

    // def forward(self, x, all_decoder_outputs):  # batch_size x max_len x input_size, batch_size x max_len x output_size
    //     max_target_length = x.shape[1]

    //     # add neutral position vector?
    //     # Run through mixture_model one time step at a time
    //     for t in range(max_target_length-1):
    //         decoder_output = x[:, t].clone().detach()
    //         mixture_merged = self.forward_step(decoder_output)
    //         all_decoder_outputs[:, t] = mixture_merged  # Store this step's outputs
    //     return all_decoder_outputs

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let x = input
        // Processing gaussian mixture params:
        let mixtureMeans = linearMixtureMeans(x)
        let mixtureVars = softplus(linearMixtureVars(x))
        let mixtureWeights = softmax(linearMixtureWeights(x), alongAxis: 1)
        // stop
        let stop = sigmoid(linearStop(x))
        // merge
        let mixtureMerged = Tensor(concatenating: [mixtureMeans, mixtureVars, mixtureWeights, stop])
        return mixtureMerged
    }

    static func getOutputSize(nbJoints: Int, nbMixtures: Int) -> Int {
        return 2 * nbMixtures * nbJoints + nbMixtures + 1
    }
}
