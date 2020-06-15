import ModelSupport
import TensorFlow
import Foundation


public struct Prediction {
    public let classIdx: Int
    public let className: String
    public let probability: Float
}

extension BERTClassifier {
    // TODO: get num_best preds
    public func predict(_ texts: [String], maxSequenceLength: Int, labels: [String], batchSize: Int) -> [Prediction] {
        print("predict()")
        print("texts: \(texts.count)")

        let validationExamples = texts.map {
            (text) -> TextBatch in
            return self.bert.preprocess(
                sequences: [text],
                maxSequenceLength: maxSequenceLength
            )
        }
        
        print("validationExamples.count: \(validationExamples.count)")

        print("batchSize: \(batchSize)")
        print("maxSequenceLength: \(maxSequenceLength)")
        print("batchSize / maxSequenceLength: \(batchSize / maxSequenceLength)")

        let validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength).map { 
            $0.paddedAndCollated(to: maxSequenceLength)
        }
        print("validationBatches: \(validationBatches.count)")
        var preds: [Prediction] = []
        for batch in validationBatches {
            print("batch")
            let logits = self(batch)
            let probs = softmax(logits, alongAxis: 1)
            let classIdxs = logits.argmax(squeezingAxis: 1)
            let batchPreds = (0..<classIdxs.shape[0]).map { 
                (idx) -> Prediction in
                let classIdx: Int = Int(classIdxs[idx].scalar!)
                let prob = probs[idx, classIdx].scalar!
                return Prediction(classIdx: classIdx, className: labels[classIdx], probability: prob)
            }
            preds.append(contentsOf: batchPreds)
        }
        return preds
    }
}
