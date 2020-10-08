import TensorFlow

public struct MotionGaussianMixtureModel: Module {

    @noDerivative public var inputSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int

    public var linearMixtureMeans: Dense<Float>
    public var linearMixtureVars: Dense<Float>
    public var linearMixtureWeights: Dense<Float>
    public var linearStop: Dense<Float>

    public var linearMixtureMeansTD: TimeDistributed
    public var linearMixtureVarsTD: TimeDistributed
    public var linearMixtureWeightsTD: TimeDistributed
    public var linearStopTD: TimeDistributed

    public init(inputSize: Int, nbJoints: Int, nbMixtures: Int) {
        self.inputSize = inputSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures

        // FC layers for learning gaussian mixture distributions params
        // input_size  // hidden_size*n_layers, FIXME: Plappert concatenates output of all rnn layers
        linearMixtureMeans = Dense<Float>(inputSize: inputSize, outputSize: nbJoints * nbMixtures)
        linearMixtureVars = Dense<Float>(inputSize: inputSize, outputSize: nbJoints * nbMixtures)
        linearMixtureWeights = Dense<Float>(inputSize: inputSize, outputSize: nbMixtures)

        // and stop bit
        linearStop = Dense<Float>(inputSize: inputSize, outputSize: 1)

        linearMixtureMeansTD = TimeDistributed(linearMixtureMeans)
        linearMixtureVarsTD = TimeDistributed(linearMixtureVars)
        linearMixtureWeightsTD = TimeDistributed(linearMixtureWeights)
        linearStopTD = TimeDistributed(linearStop)
    }

    public init(
        inputSize: Int, nbJoints: Int, nbMixtures: Int,
        linearMixtureMeans: Dense<Float>, linearMixtureVars: Dense<Float>,
        linearMixtureWeights: Dense<Float>, linearStop: Dense<Float>
    ) {
        self.inputSize = inputSize
        self.nbJoints = nbJoints
        self.nbMixtures = nbMixtures
        self.linearMixtureMeans = linearMixtureMeans
        self.linearMixtureVars = linearMixtureVars
        self.linearMixtureWeights = linearMixtureWeights
        self.linearStop = linearStop
        self.linearMixtureMeansTD = TimeDistributed(linearMixtureMeans)
        self.linearMixtureVarsTD = TimeDistributed(linearMixtureVars)
        self.linearMixtureWeightsTD = TimeDistributed(linearMixtureWeights)
        self.linearStopTD = TimeDistributed(linearStop)
    }

    @differentiable
    func forwardStep(_ x: Tensor<Float>) -> MixtureModelPreds {
        // bs x input_size
        // Processing gaussian mixture params:
        let mixtureMeans = linearMixtureMeans(x)
        let mixtureVars = softplus(linearMixtureVars(x))
        var mixtureWeights = softmax(linearMixtureWeights(x), alongAxis: 1)

        if mixtureWeights.isNaN.any() {
            // print("Fixing NaNs")
            var divider = 1.0
            let double_x = Tensor<Double>(linearMixtureWeights(x))
            while mixtureWeights.isNaN.any() {
                mixtureWeights = Tensor<Float>(softmax(double_x/divider, alongAxis: 1))
                divider *= 10.0
            }
        }

        // stop
        let stop = sigmoid(linearStop(x))
        // merge
        let mixtureStepPreds = MixtureModelPreds(mixtureMeans: mixtureMeans, mixtureVars: mixtureVars, mixtureWeights: mixtureWeights, stops: stop)
        return mixtureStepPreds
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> MixtureModelPreds {
        let targetLength = input.shape[1]
        // TODO: use time distributed layers
        // Run through mixture_model one time step at a time
        var all_outputs: [MixtureModelPreds] = []
        for t in 0..<targetLength {
            let decoder_input: Tensor<Float> = input[0..., t]
            let decoder_output = self.forwardStep(decoder_input)
            all_outputs.append(decoder_output)
        }
        
        let all_outputs_struct = MixtureModelPreds(stacking: all_outputs, alongAxis: 1)
        return all_outputs_struct
    }

    public func callAsFunction2(_ input: Tensor<Float>) -> MixtureModelPreds {
        // let targetLength = input.shape[1]

        // // Run through mixture_model one time step at a time
        // var all_outputs: [MixtureModelPreds] = []
        // for t in 0..<targetLength {
        //     let decoder_input: Tensor<Float> = input[0..., t]
        //     let decoder_output = self.forwardStep(decoder_input)
        //     all_outputs.append(decoder_output)
        // }

        let t1 = Tensor<Float>([1.0])

        let mixtureMeans = timeDistributed(input, linearMixtureMeans.weight)
        // TODO: fix mixtureWeights NaNs
        let mixtureVars = softplus(timeDistributed(input, linearMixtureVars.weight))
        let mixtureWeights = softmax(timeDistributed(input, linearMixtureWeights.weight), alongAxis: 1)
        let stops = sigmoid(timeDistributed(input, linearStop.weight))
        // print("stops.shape: \(stops.shape)")

        // let logits = timeDistributed(h, embedding.embeddings.transposed())

        // let all_outputs_struct = MixtureModelPreds(stacking: all_outputs, alongAxis: 1)
        // return all_outputs_struct
        return MixtureModelPreds(mixtureMeans: mixtureMeans, mixtureVars: mixtureVars, mixtureWeights: mixtureWeights, stops: stops)
    }
}
