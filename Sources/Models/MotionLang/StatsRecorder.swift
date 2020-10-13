import Foundation
import TrainingLoop
import x10_optimizers_optimizer
import SummaryWriter

public class StatsRecorder {
    let summaryWriter: SummaryWriter
    public var trainingStepCount = 0
    public var trainingBatchCount = 0
    public var trainingLossSum: Float = 0.0

    public var validationStepCount = 0
    public var validationBatchCount = 0
    public var validationLossSum: Float = 0.0

    public var inValidationPhase = false

    public init(logdirURL: URL) {
        summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)
    }
    
    public func writeStats<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent, model: Any) throws {
                
        if event == .validationStart {
            inValidationPhase = true
            validationBatchCount = 0
            validationLossSum = 0.0
        }

        if event == .validationEnd {
            guard let epochIndex = loop.epochIndex else {
                return
            }
            inValidationPhase = false
            let current_epoch = epochIndex + 1
            let epochValidationLoss = validationLossSum / Float(validationBatchCount)
            summaryWriter.writeScalarSummary(tag: "EpochValidationLoss", step: current_epoch, value: epochValidationLoss)
        }

        if event == .batchEnd {
            guard let lastLoss = loop.lastLoss else {
                return
            }
            if !inValidationPhase {
                summaryWriter.writeScalarSummary(tag: "TrainingLoss", step: trainingStepCount, value: lastLoss.scalar!)
                
                let optimizer: GeneralOptimizer<MotionLangTransformer> = loop.optimizer as! GeneralOptimizer<MotionLangTransformer>
                summaryWriter.writeScalarSummary(tag: "LearningRate", step: trainingStepCount, value: optimizer.learningRate)

                trainingStepCount += 1
                trainingBatchCount += 1
                trainingLossSum += Float(lastLoss.scalar!)
            } else {
                validationStepCount += 1
                validationBatchCount += 1
                validationLossSum += Float(lastLoss.scalar!)
                summaryWriter.writeScalarSummary(tag: "ValidationLoss", step: validationStepCount, value: lastLoss.scalar!)
            }
        }
        if event == .trainingStart {
            trainingBatchCount = 0
            trainingLossSum = 0.0
        }
        if event == .trainingEnd {
            guard let epochIndex = loop.epochIndex else {
                return
            }
            let current_epoch = epochIndex + 1
            let epochTrainingLoss = trainingLossSum / Float(trainingBatchCount)
            summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: current_epoch, value: epochTrainingLoss)
        }
        if event == .epochEnd || event == .fitEnd {
            summaryWriter.flush()
        }
    }
}
