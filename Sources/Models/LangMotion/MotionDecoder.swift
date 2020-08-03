import TensorFlow
import PythonKit

let np  = Python.import("numpy")

public func randomNumber(probabilities: [Double]) -> Int {
    // https://stackoverflow.com/questions/30309556/generate-random-numbers-with-a-given-distribution
    // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
    let sum = probabilities.reduce(0, +)
    // Random number in the range 0.0 <= rnd < sum :
    let rnd = Double.random(in: 0.0 ..< sum)
    // Find the first interval of accumulated probabilities into which `rnd` falls:
    var accum = 0.0
    for (i, p) in probabilities.enumerated() {
        accum += p
        if rnd < accum {
            return i
        }
    }
    // This point might be reached due to floating point inaccuracies:
    return (probabilities.count - 1)
}

public func gaussian_pdf(sample: Tensor<Float>, means: Tensor<Float>, variances: Tensor<Float>) -> Tensor<Float> {
    // one-dim tensors
    assert(sample.shape.count == 1)
    assert(sample.shape == means.shape)
    assert(sample.shape == variances.shape)
    let a1 = sqrt(Float(2.0) * Float(np.pi)! * variances)
    let a2 = -(sample - means).squared()
    return Float(1.0) / a1 * exp(a2 / (2.0 * variances))
}

public func bernoulli_pdf(sample: Int, p: Float) -> Float {
    let fSample = Float(sample)
    return fSample * p + Float(1.0 - fSample) * (1.0 - p)
}

public func performNormalMixtureSampling(preds: MixtureModelPreds, nb_joints: Int, nb_mixtures: Int, maxMotionLength: Int) -> (motion: Tensor<Float>, log_probs: [Float], done: Tensor<Int32>) {
    let TINY: Float = 1e-8
    let motionLength = preds.mixtureMeans.shape[1]

    var motion: Tensor<Float> = Tensor<Float>(zeros: [motionLength-1, nb_joints])
    var log_probs: [Float] = [Float](repeating:0.0, count: motionLength-1)
    var done: [Int32] = [Int32](repeating: 0, count: motionLength-1)

    let all_means = preds.mixtureMeans.squeezingShape(at: 0)
    let all_variances = preds.mixtureVars.squeezingShape(at: 0) + TINY
    let weights = preds.mixtureWeights.squeezingShape(at: 0)
    let stops = preds.stops[0, 0..., 0]

    /// Sample joint values.
    var samples = Tensor<Float>(zeros: [motionLength, nb_joints])
    var means = Tensor<Float>(zeros: [motionLength, nb_joints])
    var variances = Tensor<Float>(zeros: [motionLength, nb_joints])
    for width_idx in 0..<motionLength-1 { // FIXME: why -1?
        // Decide which mixture to sample from
        let p = weights[width_idx].scalars.map { Double($0)}
        assert(p.count == nb_mixtures)
        let mixture_idx = randomNumber(probabilities: p) //np.random.choice(range(nb_mixtures), p=p)

        /// Sample from it.
        let start_idx = mixture_idx * nb_joints
        let m = all_means[width_idx, start_idx..<start_idx + nb_joints]
        let v = all_variances[width_idx, start_idx..<start_idx + nb_joints]
        assert(m.shape == [nb_joints])
        assert(m.shape == v.shape)
        // https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        let s = np.random.normal(m.scalars, v.scalars)
        samples[width_idx, 0...] = Tensor<Float>(Array(s)!)
        means[width_idx, 0...] = m
        variances[width_idx, 0...] = v
    }

    for idx in 0..<samples.shape[0]-1 {
        let sample = samples[idx]
        let stop: Float = stops[idx].scalar!
        if done[idx] != 0 {
            continue
        }
        // https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html
        let sampled_stop: Int = Int(np.random.binomial(n: 1, p: stop))!
        let combined = Tensor<Float>(concatenating: [sample[0..<nb_joints-1], Tensor<Float>([Float(sampled_stop)])])
        assert(combined.shape == [nb_joints])
        motion[idx] = combined
        log_probs[idx] += log(gaussian_pdf(sample: sample, means: means[idx], variances: variances[idx])).sum().scalar!
        log_probs[idx] += log(bernoulli_pdf(sample: sampled_stop, p: stop))
        done[idx] = (sampled_stop == 0) ? 1 : 0
    }
    return (motion: motion, log_probs: log_probs, done: Tensor(done))
}

public func decode(context: Any, nb_joints: Int, language: Any, references: Any, args: Any, init: Any? = nil) {
// def decode(context, nb_joints, language, references, args, init=None):
    // # Prepare data structures for graph search.
    // if init is None:
    //     init = np.ones(nb_joints + 1)
    // assert init.shape == (nb_joints + 1,)
    // previous_outputs = [[np.copy(init)] for _ in range(args.width)]
    // repeated_context = np.repeat(context.reshape(
    //     1, context.shape[-1]), args.width, axis=0)
    // repeated_context = repeated_context.reshape(
    //     args.width, 1, context.shape[-1])
    // log_probabilities = [0. for _ in range(args.width)]
    // done = [False for _ in range(args.width)]

    // # Reset the decoder.
    // v_decoder_hidden = Variable(torch.zeros(decoder.n_layers, args.width, decoder.dec_hidden_size, dtype=torch.float32))

    // # Iterate over time.
    // predictions = [[] for _ in range(args.width)]

    // for _ in range(args.depth):
    //     previous_output = np.array([o[-1] for o in previous_outputs])
    //     assert previous_output.ndim == 2
    //     previous_output = previous_output.reshape(
    //         (previous_output.shape[0], 1, previous_output.shape[1]))
    //     t_repeated_context = torch.Tensor(repeated_context.squeeze(axis=1))
    //     t_previous_output = torch.Tensor(previous_output.squeeze(axis=1))
    //     t_encoder_outputs = torch.Tensor(encoder_outputs.detach().numpy().repeat(args.width, 1))
        // decoder_output, v_decoder_hidden, decoder_attn = \
        //     decoder(t_repeated_context, t_previous_output, v_decoder_hidden, t_encoder_outputs)

        // decoder_output = decoder.predict_on_batch([repeated_context, previous_output])

        // preds = np.expand_dims(decoder_output.detach().numpy(), axis=1)
    //     assert preds.shape[0] == args.width
    //     for idx, (pred, d) in enumerate(zip(preds, done)):
    //         if d:
    //             continue
    //         predictions[idx].append(pred)

        // Perform actual decoding.
        // let (previous_outputs, log_probabilities, done) = performNormalMixtureSampling(
        //     preds: preds, nb_joints: nb_joints, nb_mixtures: nb_joints, maxMotionLength: 50)

    //     if args.motion_representation == 'hybrid':
    //         # For each element of the beam, add the new delta (index -1) to the previous element (index -2)
    //         # to obtain the absolute motion.
    //         for po in previous_outputs:
    //             po[-1][:nb_joints] = po[-2][:nb_joints] + po[-1][:nb_joints]
        
    //     # Check if we're done before reaching `args.depth`.
    //     if np.all(done):
    //         break

    // # Convert to numpy arrays.
    // predictions = [np.array(preds)[:, 0, :].astype('float32')
    //                for preds in predictions]

    // hypotheses = []
    // for previous_output in previous_outputs:
    //     motion = np.array(previous_output)[1:].astype(
    //         'float32')  # remove init state
    //     if args.motion_representation == 'diff':
    //         motion[:, :nb_joints] = np.cumsum(motion[:, :nb_joints], axis=0)
    //     assert motion.shape[-1] == nb_joints + 1
    //     hypotheses.append(motion.astype('float32'))

    // # Record data.
    // data = {
    //     'hypotheses': hypotheses,
    //     'log_probabilities': log_probabilities,
    //     'references': references,
    //     'language': language,
    //     'predictions': predictions,
    // }
    // return data
}