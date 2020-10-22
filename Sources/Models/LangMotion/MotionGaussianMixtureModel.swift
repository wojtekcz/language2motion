import TensorFlow

public struct MotionGaussianMixtureModel: Module {

    @noDerivative public var inputSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var nbMixtures: Int

    public var linearMixtureMeans: Dense<Float>
    public var linearMixtureVars: Dense<Float>
    public var linearMixtureWeights: Dense<Float>
    public var linearStop: Dense<Float>

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
    }

//    @differentiable
//    func fixMixtureWeightsStep(_ x: Tensor<Float>) -> Tensor<Float> {
//        // x should be 1 x input_size
//        //var x = x.expandingShape(at: 0)
//        print("x.shape: \(x.shape)")
//        let y = linearMixtureWeights(x)
//        print("y.shape: \(y.shape)")
//        var mixtureWeights = softmax(y, alongAxis: 1)
//
//        if mixtureWeights.isNaN.any() {
////            print("\nFixing NaNs")
//            var divider = 1.0
//            let double_x = Tensor<Double>(linearMixtureWeights(x))
//            var count = 5
//            while mixtureWeights.isNaN.any() && count>0 {
//                let y = double_x/divider
//                print("count: \(count), y.shape: \(y.shape)")
//                mixtureWeights = Tensor<Float>(softmax(y, alongAxis: 1))
//                divider *= 10.0
//                count = count - 1
//            }
//        }
//        return mixtureWeights
//    }

//    @differentiable
//    func fixMixtureWeights(_ input: Tensor<Float>) -> Tensor<Float> {
//        print("input.shape: \(input.shape)")
//        let bs = input.shape[0]
//        let targetLength = input.shape[1]
//        // Run through mixture_model one time step at a time
//        var all_outputs: [Tensor<Float>] = []
//        for s in 0..<bs {
//            var sample_outputs: [Tensor<Float>] = []
//            for t in 0..<targetLength {
//                let decoder_input: Tensor<Float> = input[s, t].expandingShape(at: 0)
//                print("s: \(s), t: \(t), decoder_input.shape: \(decoder_input.shape)")
//                let decoder_output = self.fixMixtureWeightsStep(decoder_input)
//                sample_outputs.append(decoder_output)
//            }
//            let sample_outputs_struct = Tensor(stacking: sample_outputs, alongAxis: 1).squeezingShape(at: 0)
//            all_outputs.append(sample_outputs_struct)
//        }
//        let all_outputs_struct = Tensor(stacking: all_outputs, alongAxis: 0)
//        return all_outputs_struct
//    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> MixtureModelPreds {
        let mixtureMeans = timeDistributed(input, linearMixtureMeans.weight)
        let mixtureVars = softplus(timeDistributed(input, linearMixtureVars.weight))
        let mixtureWeights =  softmax(timeDistributed(input, linearMixtureWeights.weight), alongAxis: 2)
//        if mixtureWeights.isNaN.any() {
//            print("\nFixing NaNs")
//            mixtureWeights = fixMixtureWeights(input)
//        }
        let stops = sigmoid(timeDistributed(input, linearStop.weight))
        return MixtureModelPreds(mixtureMeans: mixtureMeans, mixtureVars: mixtureVars, mixtureWeights: mixtureWeights, stops: stops)
    }
}
