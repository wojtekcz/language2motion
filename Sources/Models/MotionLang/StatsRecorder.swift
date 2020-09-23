import Foundation
import TrainingLoop
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

    public var epochIndex = 0 // FIXME: Workaround
    public var inValidationPhase = false

    public init(logdirURL: URL) {
        summaryWriter = SummaryWriter(logdir: logdirURL, flushMillis: 30*1000)
    }
    
    public func writeStats<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
        
        // TODO: write learning rate
        
        if event == .validationStart {
            inValidationPhase = true
            validationBatchCount = 0
            validationLossSum = 0.0
        }

        if event == .validationEnd {
            inValidationPhase = false
            let current_epoch = epochIndex + 1
            let epochValidationLoss = validationLossSum / Float(validationBatchCount)
            summaryWriter.writeScalarSummary(tag: "EpochValidationLoss", step: current_epoch, value: epochValidationLoss)
        }

        if event == .batchEnd {
            guard
            // let batchIndex = loop.batchIndex,
            let lastLoss = loop.lastLoss else {
                return
            }
            if !inValidationPhase {
                // print("\nbatch stats: batchIndex: \(batchIndex), trainingStepCount: \(trainingStepCount), trainingLoss: \(lastLoss)")
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
            // guard let epochIndex = loop.epochIndex else {
            //     return
            // }
            let current_epoch = epochIndex + 1
            let epochTrainingLoss = trainingLossSum / Float(trainingBatchCount)
            // print("\nepoch stats: current_epoch: \(current_epoch), epochTrainingLoss: \(epochTrainingLoss)")
            summaryWriter.writeScalarSummary(tag: "EpochTrainingLoss", step: current_epoch, value: epochTrainingLoss)
        }
        if event == .fitEnd {
            summaryWriter.flush()
        }
    }
}
