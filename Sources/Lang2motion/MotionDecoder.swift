import TensorFlow
import PythonKit

let np  = Python.import("numpy")

func randomNumber(probabilities: [Double]) -> Int {
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

// def gaussian_pdf(sample, means, variances):
//     assert sample.ndim == 1
//     assert sample.shape == means.shape
//     assert sample.shape == variances.shape

//     return 1. / (np.sqrt(2. * np.pi * variances)) * np.exp(-np.square(sample - means) / (2. * variances))
func gaussian_pdf(sample: Float, means: Float, variances: Float) -> Float {
    // let a1 = np.sqrt(2.0 * np.pi * variances)
    // let a2 = np.exp(-np.square(sample - means)
    // return Float(1.0) / a1 * a2 / (2.0 * variances)
    return 0.0
}


// def bernoulli_pdf(sample, p):
//     return float(sample) * p + float(1. - sample) * (1. - p)
func bernoulli_pdf(sample: Float, p: Float) -> Float {
    return 0.0
}

    // let b_pdf1 = Float(1.0) - y_true_stop
    // let b_pdf2 = Float(1.0) - stop
    // let bernoulli_pdf = y_true_stop * stop + b_pdf1 * b_pdf2


func perform_normal_mixture_sampling(preds: Tensor<Float>, decoder: Any, nb_joints: Int, 
                                     previous_outputs: [[Tensor<Float>]], log_probabilities: [[Float]],
                                     done: [Bool], nb_mixtures: Int) -> ([[Tensor<Float>]], [[Float]], [Bool]) {
// def perform_normal_mixture_sampling(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args):
    let TINY: Float = 1e-8
    let _preds = preds.reshaped(to:
        [preds.shape[0], 2 * nb_joints * nb_mixtures + nb_mixtures + 1])
    let all_means = _preds[0..., 0..<nb_joints * nb_mixtures]
    let all_variances = _preds[0..., nb_joints *
                              nb_mixtures..<2 * nb_joints * nb_mixtures] + TINY
    let weights = _preds[0..., 2 * nb_joints * nb_mixtures..<2 *
                        nb_joints * nb_mixtures + nb_mixtures]
    assert(all_means.shape[-1] == nb_joints * nb_mixtures)
    assert(all_variances.shape[-1] == nb_joints * nb_mixtures)
    assert(weights.shape[-1] == nb_mixtures)
    let stops = _preds[0..., -1]

    /// Sample joint values.
    var samples = Tensor<Float>(zeros: [_preds.shape[0], nb_joints])
    var means = Tensor<Float>(zeros: [_preds.shape[0], nb_joints])
    var variances = Tensor<Float>(zeros: [_preds.shape[0], nb_joints])
    for width_idx in 0..<_preds.shape[0] {
        // Decide which mixture to sample from
        let p = weights[width_idx].scalars.map { Double($0)}
        // assert p.shape == (nb_mixtures,)
        let mixture_idx = randomNumber(probabilities: p) //np.random.choice(range(nb_mixtures), p=p)

        /// Sample from it.
        let start_idx = mixture_idx * nb_joints
        let m = all_means[width_idx, start_idx..<start_idx + nb_joints]
        let v = all_variances[width_idx, start_idx..<start_idx + nb_joints]
        assert(m.shape == [nb_joints])
        assert(m.shape == v.shape)
        // https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        let s = np.random.normal(m.scalars, v.scalars)
        samples[width_idx, 0...] = Tensor(numpy: s)!
        means[width_idx, 0...] = m
        variances[width_idx, 0...] = v
    }

    var _previous_outputs = previous_outputs
    // for idx, (sample, stop) in enumerate(zip(samples, stops)):
    for idx in 0..<samples.shape[0] {
        let sample = samples[idx]
        let stop = stops[idx]
        if done[idx] {
            continue
        }
        // https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html
        let sampled_stop = np.random.binomial(n: 1, p: stop.scalars)
        let combined = Tensor<Float>(concatenating: [sample, Tensor<Float>(numpy: sampled_stop)!])
        assert(combined.shape == [nb_joints + 1])
        _previous_outputs[idx].append(combined)
        // let a2 = np.log(gaussian_pdf(sample.scalars, means[idx], variances[idx]))
        // let a1: Float = np.sum(a2)

        // log_probabilities[idx] += a1
        // log_probabilities[idx] += np.log(bernoulli_pdf(sampled_stop, stop))
        // done[idx] = (sampled_stop == 0)
    }
    return (_previous_outputs, log_probabilities, done)
}

func decode(context: Any, nb_joints: Int, language: Any, references: Any, args: Any, init: Any? = nil) {
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
    //     decoder_output, v_decoder_hidden, decoder_attn = \
    //         decoder(t_repeated_context, t_previous_output, v_decoder_hidden, t_encoder_outputs)
    //     preds = np.expand_dims(decoder_output.detach().numpy(), axis=1)
    //     assert preds.shape[0] == args.width
    //     for idx, (pred, d) in enumerate(zip(preds, done)):
    //         if d:
    //             continue
    //         predictions[idx].append(pred)

    //     # Perform actual decoding.
    //     if args.decoder == 'normal':
    //         fn = perform_normal_sampling
    //     elif args.decoder == 'regression':
    //         fn = perform_regression
    //     elif args.decoder == 'normal-mixture':
    //         fn = perform_normal_mixture_sampling
    //     else:
    //         fn = None
    //         raise ValueError('Unknown decoder "{}"'.format(args.decoder))
    //     previous_outputs, log_probabilities, done = fn(
    //         preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args)

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