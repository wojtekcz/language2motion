import TensorFlow
import Datasets

public struct LossArgs {
    public let nb_joints: Int
    public let nb_mixtures: Int
    public let mixture_regularizer_type: String  // ["cv", "l2", "None"]
    public let mixture_regularizer: Float
    public let device: Device

    public init(nb_joints: Int, nb_mixtures: Int, mixture_regularizer_type: String, mixture_regularizer: Float, device: Device) {
        self.nb_joints = nb_joints
        self.nb_mixtures = nb_mixtures
        self.mixture_regularizer_type = mixture_regularizer_type
        self.mixture_regularizer = mixture_regularizer
        self.device = device
     }
}

@differentiable(wrt: y_pred)
public func _normalMixtureSurrogateLoss(y_true: LangMotionBatch.Target, y_pred: MixtureModelPreds, args: LossArgs) -> Tensor<Float> {
    let TINY: Float = 1e-8
    let pi: Float = 3.1415
    let nb_mixtures = args.nb_mixtures
    let nb_joints = args.nb_joints

    let all_means = y_pred.mixtureMeans
    let all_variances = y_pred.mixtureVars + TINY
    let weights = y_pred.mixtureWeights
    let stops = y_pred.stops.squeezingShape(at: 2)

    var log_mixture_pdf: Tensor<Float> = Tensor<Float>(zeros: [weights.shape[0], weights.shape[1]], on: args.device) 
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
    
    // mask mixture loss with stops
    let zeroTensor = Tensor<Float>(repeating: 0.0, shape: log_mixture_pdf.shape, on: args.device)
    log_mixture_pdf = log_mixture_pdf.replacing(with: zeroTensor, where: y_true.stops .== Tensor<Float>(1.0, on: args.device))

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
    let loss = -(log_mixture_pdf + log_bernoulli_pdf) +
        args.mixture_regularizer * mixture_reg
    return loss
}

extension LangMotionBatch.Target {

    public func squeezed() -> Self {
        let bs = self.motion.shape[0]
        let nFrames = self.motion.shape[1]
        let nJoints = self.motion.shape[2]
        let motion = self.motion.reshaped(to: [1, bs*nFrames, nJoints])
        let stops = self.stops.reshaped(to: [1, bs*nFrames])
        let segmentIDs = self.segmentIDs.reshaped(to: [1, bs*nFrames])
        let origMotionFramesCount = self.origMotionFramesCount.sum().expandingShape(at: 0)
        return Self(sampleID: self.sampleID, motion: motion, stops: stops, segmentIDs: segmentIDs, origMotionFramesCount: origMotionFramesCount)
    }

    public func gathering(atIndices indices: Tensor<Int32>, alongAxis axis: Int) -> Self {
        let motion = self.motion.gathering(atIndices: indices, alongAxis: axis)
        let stops = self.stops.gathering(atIndices: indices, alongAxis: axis)
        let segmentIDs = self.segmentIDs.gathering(atIndices: indices, alongAxis: axis)
        return Self(sampleID: self.sampleID, motion: motion, stops: stops, segmentIDs: segmentIDs, origMotionFramesCount: self.origMotionFramesCount)
    }
}

extension MixtureModelPreds {
    
    @differentiable
    public func squeezed() -> Self {
        let bs = self.mixtureMeans.shape[0]
        let nFrames = self.mixtureMeans.shape[1]
        let nJointsMixtures = self.mixtureMeans.shape[2]
        let nMixtures = self.mixtureWeights.shape[2]
        
        let mixtureMeans = self.mixtureMeans.reshaped(to: [1, bs*nFrames, nJointsMixtures])
        let mixtureVars = self.mixtureVars.reshaped(to: [1, bs*nFrames, nJointsMixtures])
        let mixtureWeights = self.mixtureWeights.reshaped(to: [1, bs*nFrames, nMixtures])
        let stops = self.stops.reshaped(to: [1, bs*nFrames, 1])        
        
        return Self(mixtureMeans: mixtureMeans, mixtureVars: mixtureVars, mixtureWeights: mixtureWeights, stops: stops)
    }
    
    @differentiable
    public func gathering(atIndices indices: Tensor<Int32>, alongAxis axis: Int) -> Self {
        let maskedMixtureMeans = self.mixtureMeans.gathering(atIndices: indices, alongAxis: axis)
        let maskedMixtureVars = self.mixtureVars.gathering(atIndices: indices, alongAxis: axis)
        let maskedMixtureWeights = self.mixtureWeights.gathering(atIndices: indices, alongAxis: axis)
        let maskedStops = self.stops.gathering(atIndices: indices, alongAxis: axis)
        return Self(mixtureMeans: maskedMixtureMeans, mixtureVars: maskedMixtureVars, mixtureWeights: maskedMixtureWeights, stops: maskedStops)
    }
}

@differentiable(wrt: y_pred)
public func normalMixtureSurrogateLoss(y_pred: MixtureModelPreds, y_true: LangMotionBatch.Target, args: LossArgs) -> Tensor<Float> {
    // masking
    var y_pred = y_pred.squeezed()
    var y_true = y_true.squeezed()
    let ids = Tensor<Int32>(rangeFrom: 0, to: Int32(y_true.stops.shape[1]), stride: 1, on: args.device)
    let indices = ids.gathering(where: y_true.segmentIDs .!= Tensor(0, on: args.device))
    y_pred = y_pred.gathering(atIndices: indices, alongAxis: 1)
    y_true = y_true.gathering(atIndices: indices, alongAxis: 1)
    
    let loss = _normalMixtureSurrogateLoss(y_true: y_true, y_pred: y_pred, args: args)    
    let mean_loss = loss.mean()
    return mean_loss
}
