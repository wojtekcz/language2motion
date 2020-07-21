// def perform_normal_mixture_sampling(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args):
//     TINY = 1e-8
//     preds = preds.reshape(
//         preds.shape[0], 2 * nb_joints * args.nb_mixtures + args.nb_mixtures + 1)
//     all_means = preds[:, :nb_joints * args.nb_mixtures]
//     all_variances = preds[:, nb_joints *
//                           args.nb_mixtures:2 * nb_joints * args.nb_mixtures] + TINY
//     weights = preds[:, 2 * nb_joints * args.nb_mixtures:2 *
//                     nb_joints * args.nb_mixtures + args.nb_mixtures]
//     assert all_means.shape[-1] == nb_joints * args.nb_mixtures
//     assert all_variances.shape[-1] == nb_joints * args.nb_mixtures
//     assert weights.shape[-1] == args.nb_mixtures
//     stops = preds[:, -1]

//     # Sample joint values.
//     samples = np.zeros((preds.shape[0], nb_joints))
//     means = np.zeros((preds.shape[0], nb_joints))
//     variances = np.zeros((preds.shape[0], nb_joints))
//     for width_idx in range(preds.shape[0]):
//         # Decide which mixture to sample from
//         p = weights[width_idx]
//         assert p.shape == (args.nb_mixtures,)
//         mixture_idx = np.random.choice(range(args.nb_mixtures), p=p)

//         # Sample from it.
//         start_idx = mixture_idx * nb_joints
//         m = all_means[width_idx, start_idx:start_idx + nb_joints]
//         v = all_variances[width_idx, start_idx:start_idx + nb_joints]
//         assert m.shape == (nb_joints,)
//         assert m.shape == v.shape
//         s = np.random.normal(m, v)
//         samples[width_idx, :] = s
//         means[width_idx, :] = m
//         variances[width_idx, :] = v

//     for idx, (sample, stop) in enumerate(zip(samples, stops)):
//         if done[idx]:
//             continue

//         sampled_stop = np.random.binomial(n=1, p=stop)
//         combined = np.concatenate([sample, [sampled_stop]])
//         assert combined.shape == (nb_joints + 1,)
//         previous_outputs[idx].append(combined)
//         log_probabilities[idx] += np.sum(
//             np.log(gaussian_pdf(sample, means[idx], variances[idx])))
//         log_probabilities[idx] += np.log(bernoulli_pdf(sampled_stop, stop))
//         done[idx] = (sampled_stop == 0)

//     return previous_outputs, log_probabilities, done


// def decode(context, nb_joints, language, references, args, init=None):
//     # Prepare data structures for graph search.
//     if init is None:
//         init = np.ones(nb_joints + 1)
//     assert init.shape == (nb_joints + 1,)
//     previous_outputs = [[np.copy(init)] for _ in range(args.width)]
//     repeated_context = np.repeat(context.reshape(
//         1, context.shape[-1]), args.width, axis=0)
//     repeated_context = repeated_context.reshape(
//         args.width, 1, context.shape[-1])
//     log_probabilities = [0. for _ in range(args.width)]
//     done = [False for _ in range(args.width)]

//     # Reset the decoder.
//     v_decoder_hidden = Variable(torch.zeros(decoder.n_layers, args.width, decoder.dec_hidden_size, dtype=torch.float32))

//     # Iterate over time.
//     predictions = [[] for _ in range(args.width)]

//     for _ in range(args.depth):
//         previous_output = np.array([o[-1] for o in previous_outputs])
//         assert previous_output.ndim == 2
//         previous_output = previous_output.reshape(
//             (previous_output.shape[0], 1, previous_output.shape[1]))
//         t_repeated_context = torch.Tensor(repeated_context.squeeze(axis=1))
//         t_previous_output = torch.Tensor(previous_output.squeeze(axis=1))
//         t_encoder_outputs = torch.Tensor(encoder_outputs.detach().numpy().repeat(args.width, 1))
//         decoder_output, v_decoder_hidden, decoder_attn = \
//             decoder(t_repeated_context, t_previous_output, v_decoder_hidden, t_encoder_outputs)
//         preds = np.expand_dims(decoder_output.detach().numpy(), axis=1)
//         assert preds.shape[0] == args.width
//         for idx, (pred, d) in enumerate(zip(preds, done)):
//             if d:
//                 continue
//             predictions[idx].append(pred)

//         # Perform actual decoding.
//         if args.decoder == 'normal':
//             fn = perform_normal_sampling
//         elif args.decoder == 'regression':
//             fn = perform_regression
//         elif args.decoder == 'normal-mixture':
//             fn = perform_normal_mixture_sampling
//         else:
//             fn = None
//             raise ValueError('Unknown decoder "{}"'.format(args.decoder))
//         previous_outputs, log_probabilities, done = fn(
//             preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args)

//         if args.motion_representation == 'hybrid':
//             # For each element of the beam, add the new delta (index -1) to the previous element (index -2)
//             # to obtain the absolute motion.
//             for po in previous_outputs:
//                 po[-1][:nb_joints] = po[-2][:nb_joints] + po[-1][:nb_joints]
        
//         # Check if we're done before reaching `args.depth`.
//         if np.all(done):
//             break

//     # Convert to numpy arrays.
//     predictions = [np.array(preds)[:, 0, :].astype('float32')
//                    for preds in predictions]

//     hypotheses = []
//     for previous_output in previous_outputs:
//         motion = np.array(previous_output)[1:].astype(
//             'float32')  # remove init state
//         if args.motion_representation == 'diff':
//             motion[:, :nb_joints] = np.cumsum(motion[:, :nb_joints], axis=0)
//         assert motion.shape[-1] == nb_joints + 1
//         hypotheses.append(motion.astype('float32'))

//     # Record data.
//     data = {
//         'hypotheses': hypotheses,
//         'log_probabilities': log_probabilities,
//         'references': references,
//         'language': language,
//         'predictions': predictions,
//     }
//     return data
