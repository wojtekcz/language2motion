import Foundation
import x10_optimizers_optimizer
import TrainingLoop
import MotionLangModels
import TextModels


public struct OptimizerOpts {
    public let peakLearningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let useBiasCorrection: Bool
    
    public init(peakLearningRate: Float = 5e-4, beta1: Float = 0.9, beta2: Float = 0.999, useBiasCorrection: Bool = false) {
        self.peakLearningRate = peakLearningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.useBiasCorrection = useBiasCorrection
    }
}

public class OptimizerWrapper {
    public let opts: OptimizerOpts
    public var optimizer: GeneralOptimizer<MotionLangTransformer>
    public var scheduledLearningRate: LinearlyDecayedParameter<LinearlyWarmedUpParameter<FixedParameter<Float>>>

    public init(opts: OptimizerOpts, model: GeneralOptimizer<MotionLangTransformer>.Model) {
        self.opts = opts
        
        self.optimizer = x10_optimizers_optimizer.GeneralOptimizer(
            for: model,
            TensorVisitorPlan(model.differentiableVectorView),
            defaultOptimizer: makeWeightDecayedAdam(
                learningRate: opts.peakLearningRate,
                beta1: opts.beta1,
                beta2: opts.beta2
            )
        )
        
        self.scheduledLearningRate = LinearlyDecayedParameter(
          baseParameter: LinearlyWarmedUpParameter(
            baseParameter: FixedParameter<Float>(opts.peakLearningRate),
              warmUpStepCount: 20,
              warmUpOffset: 0),
            slope: -(opts.peakLearningRate / Float(stepsPerEpoch * nEpochs * lrSlopeMultiplier)),  // The LR decays linearly to zero.
          startStep: 10
        )
    }
    
    public func learningRateUpdater<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
        if event == .updateStart {
            let optimizer: GeneralOptimizer<MotionLangTransformer> = loop.optimizer as! GeneralOptimizer<MotionLangTransformer>
            let step = optimizer.step + 1 // for scheduled rates and bias correction, steps start at 1
            optimizer.learningRate = scheduledLearningRate(forStep: UInt64(step))
            if opts.useBiasCorrection {
              let f_step = Float(step)
                optimizer.learningRate *= sqrtf(1 - powf(opts.beta2, f_step)) / (1 - powf(opts.beta1, f_step))
            }
            // print("\noptimizer: step: \(optimizer.step), learningRate: \(optimizer.learningRate)")
        }
    }
}
