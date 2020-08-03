import TensorFlow

public struct LossArgs {
    public let nb_joints: Int
    public let nb_mixtures: Int
    public let mixture_regularizer_type: String  // ["cv", "l2", "None"]
    public let mixture_regularizer: Float

    public init(nb_joints: Int, nb_mixtures: Int, mixture_regularizer_type: String, mixture_regularizer: Float) {
        self.nb_joints = nb_joints
        self.nb_mixtures = nb_mixtures
        self.mixture_regularizer_type = mixture_regularizer_type
        self.mixture_regularizer = mixture_regularizer
     }
}

public struct TargetTruth {
    public let motion: Tensor<Float> // bs x maxMotionLength-1 x nbJoints
    public let stops: Tensor<Float>  // bs x maxMotionLength-1

    public init(motion: Tensor<Float>, stops: Tensor<Float>) {
        self.motion = motion
        self.stops = stops
     }
}

@differentiable
public func normalMixtureSurrogateLoss(y_true: TargetTruth, y_pred: MixtureModelPreds, args: LossArgs) -> Tensor<Float> {
    let TINY: Float = 1e-8
    let pi: Float = 3.1415
    let nb_mixtures = args.nb_mixtures
    let nb_joints = args.nb_joints

    let all_means = y_pred.mixtureMeans
    let all_variances = y_pred.mixtureVars + TINY
    let weights = y_pred.mixtureWeights
    let stops = y_pred.stops.squeezingShape(at: 2)

    var log_mixture_pdf: Tensor<Float> = Tensor<Float>(zeros: [weights.shape[0], weights.shape[1]]) 
    for mixture_idx in 0..<nb_mixtures {
        let start_idx = mixture_idx * nb_joints
        let means = all_means[0..., 0..., start_idx..<start_idx + nb_joints]
        let variances = all_variances[0..., 0..., start_idx..<start_idx + nb_joints]
        let diff = y_true.motion - means
        let pdf1 = 1.0 / sqrt(variances * 2.0 * pi)
        let pdf2a = diff.squared()
        let pdf2 = exp(-(pdf2a) / (2.0 * variances))
        let pdf = pdf1 * pdf2
        let weighted_pdf = weights[0..., 0..., mixture_idx] * 
            log(pdf + TINY).sum(alongAxes:2).squeezingShape(at: 2)
        log_mixture_pdf = log_mixture_pdf + weighted_pdf
    }

    let b_pdf1 = Float(1.0) - y_true.stops
    let b_pdf2 = Float(1.0) - stops
    let bernoulli_pdf = y_true.stops * stops + b_pdf1 * b_pdf2
    let log_bernoulli_pdf = log(bernoulli_pdf + TINY)

    var mixture_reg: Float = 0.0
    if args.mixture_regularizer_type == "cv" {
        // We want to use (std / mean)^2 = std^2 / mean^2 = var / mean^2.
        mixture_reg = weights.variance().scalarized() / 
            weights.mean().squared().scalarized()
    } else if args.mixture_regularizer_type == "l2" {
        mixture_reg = weights.squared().sum().scalarized()
    } else {
        mixture_reg = 0.0
    }
    // TODO: divide loss (component?) by maxMotionLength
    // TODO: move loss averaging here
    let loss = -(log_mixture_pdf + log_bernoulli_pdf) +
        args.mixture_regularizer * mixture_reg
    return loss
}
