// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(
    fileURLWithPath: "bert_models", isDirectory: true,
    relativeTo: URL(
        fileURLWithPath: NSTemporaryDirectory(),
        isDirectory: true))
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 5)

// Regarding the batch size, note that the way batching is performed currently is that we bucket
// input sequences based on their length (e.g., first bucket contains sequences of length 1 to 10,
// second 11 to 20, etc.). We then keep processing examples in the input data pipeline until a
// bucket contains enough sequences to form a batch. The batch size specified in the task
// constructor specifies the *total number of tokens in the batch* and not the total number of
// sequences. So, if the batch size is set to 1024, the first bucket (i.e., lengths 1 to 10)
// will need 1024 / 10 = 102 examples to form a batch (every sentence in the bucket is padded
// to the max length of the bucket). This kind of bucketing is common practice with NLP models and
// it is done to improve memory usage and computational efficiency when dealing with sequences of
// varied lengths. Note that this is not used in the original BERT implementation released by
// Google and so the batch size setting here is expected to differ from that one.
let maxSequenceLength = 50
let batchSize = 3096

let dsURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/labels_ds_v2.csv")

var dataset = try Language2Label(
    datasetURL: dsURL,
    maxSequenceLength: maxSequenceLength,
    batchSize: batchSize,
    entropy: SystemRandomNumberGenerator()
) { (example: Language2LabelExample) -> LabeledTextBatch in
    let textBatch = bertClassifier.bert.preprocess(
        sequences: [example.text],
        maxSequenceLength: maxSequenceLength)
   return (data: textBatch, 
           label: example.label.map { 
               (label: Language2LabelExample.LabelTuple) in Tensor(Int32(label.idx))
           }!
          )
}

print("Dataset acquired.")

var optimizer = WeightDecayedAdam(
    for: bertClassifier,
    learningRate: LinearlyDecayedParameter(
        baseParameter: LinearlyWarmedUpParameter(
            baseParameter: FixedParameter<Float>(2e-5),
            warmUpStepCount: 10,
            warmUpOffset: 0),
        // slope: -5e-7,  // The LR decays linearly to zero in 100 steps.
        slope: -1e-7,  // The LR decays linearly to zero in ~500 steps.
        startStep: 10),
    weightDecayRate: 0.01,
    maxGradientGlobalNorm: 1)

print("Training BERT for the Language2Label task!")

time() {
    for (epoch, epochBatches) in dataset.trainingEpochs.prefix(5).enumerated() {
        print("[Epoch \(epoch + 1)]")
        Context.local.learningPhase = .training
        var trainingLossSum: Float = 0
        var trainingBatchCount = 0
        print("epochBatches.count: \(epochBatches.count)")

        for batch in epochBatches {
            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            let (loss, gradients) = valueWithGradient(at: bertClassifier) { model -> Tensor<Float> in
                let logits = model(documents)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }

            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&bertClassifier, along: gradients)

            print(
                """
                Training loss: \(trainingLossSum / Float(trainingBatchCount))
                """
            )
        }

        print("dataset.validationBatches.count: \(dataset.validationBatches.count)")
        Context.local.learningPhase = .inference
        var devLossSum: Float = 0
        var devBatchCount = 0
        var correctGuessCount = 0
        var totalGuessCount = 0

        for batch in dataset.validationBatches {
            let valBatchSize = batch.data.tokenIds.shape[0]

            let (documents, labels) = (batch.data, Tensor<Int32>(batch.label))
            let logits = bertClassifier(documents)
            let loss = softmaxCrossEntropy(logits: logits, labels: labels)
            devLossSum += loss.scalarized()
            devBatchCount += 1

            let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels

            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += valBatchSize
        }
        
        let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
        print(
            """
            Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
            Eval loss: \(devLossSum / Float(devBatchCount))
            """
        )
    }
}
