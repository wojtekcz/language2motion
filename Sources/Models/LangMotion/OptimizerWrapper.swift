import Foundation
import x10_optimizers_optimizer
import TrainingLoop
import TextModels


public struct OptimizerOpts {
    public let peakLearningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let weightDecayRate: Float
    public let useBiasCorrection: Bool
    public let lrSlopeMultiplier: Float
    public let nEpochs: Int
    public var stepsPerEpoch: Int
    public let fixedPeekLR: Bool
    
    public init(peakLearningRate: Float = 5e-4, beta1: Float = 0.9, beta2: Float = 0.999, weightDecayRate: Float, useBiasCorrection: Bool = false, lrSlopeMultiplier: Float = 1.0, nEpochs: Int = 10, stepsPerEpoch: Int = 1, fixedPeekLR: Bool = false) {
        self.peakLearningRate = peakLearningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weightDecayRate = weightDecayRate
        self.useBiasCorrection = useBiasCorrection
        self.lrSlopeMultiplier = lrSlopeMultiplier
        self.nEpochs = nEpochs
        self.stepsPerEpoch = stepsPerEpoch
        self.fixedPeekLR = fixedPeekLR
    }
}

public class OptimizerWrapper {
    public let opts: OptimizerOpts
    public var optimizer: GeneralOptimizer<LangMotionCatDistTransformer>
    public var scheduledLearningRate: LinearlyDecayedParameter<LinearlyWarmedUpParameter<FixedParameter<Float>>>

    public init(opts: OptimizerOpts, model: GeneralOptimizer<LangMotionCatDistTransformer>.Model) {
        self.opts = opts
        
        self.optimizer = x10_optimizers_optimizer.GeneralOptimizer(
            for: model,
            TensorVisitorPlan(model.differentiableVectorView),
            defaultOptimizer: makeWeightDecayedAdam(
                learningRate: opts.peakLearningRate,
                beta1: opts.beta1,
                beta2: opts.beta2,
                weightDecayRate: opts.weightDecayRate
            )
        )
        
        var slope = -(opts.peakLearningRate / Float(Float(opts.stepsPerEpoch) * Float(opts.nEpochs) * opts.lrSlopeMultiplier))  // The LR decays linearly to zero.
        
        if opts.fixedPeekLR {
            slope = 0
        }
        
        self.scheduledLearningRate = LinearlyDecayedParameter(
          baseParameter: LinearlyWarmedUpParameter(
            baseParameter: FixedParameter<Float>(opts.peakLearningRate),
              warmUpStepCount: 20*4,
              warmUpOffset: 0),
            slope: slope,
          startStep: 10*4
        )
    }
    
    public func learningRateUpdater<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: Any) throws {
        if event == .updateStart {
            let optimizer: GeneralOptimizer<LangMotionCatDistTransformer> = loop.optimizer as! GeneralOptimizer<LangMotionCatDistTransformer>
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
