//
//  DatasetManager.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 31/10/2020.
//

import Foundation
import TensorFlow
import Datasets
import ModelSupport
import TextModels


public class DatasetManager {

    #if os(macOS)
        static let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
        static let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    var dataset: Lang2Motion? = nil
    var textProcessor: TextProcessor? = nil
    var discretizer: MotionDiscretizer? = nil
    let maxTextSequenceLength =  40
    let maxMotionLength =  150
    let batchSize = 10

    func loadDataset(datasetSize: DatasetSize, device: Device) throws -> Lang2Motion {
        /// load dataset
        print("\nLoading dataset...")

        let motionDatasetURL = Self.dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

        let dataset = try Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: batchSize,
            minMotionLength: 20,
            maxMotionLength: 150,
            discretizer: &discretizer!,
            trainTestSplit: 1.0,
            device: device
        ) { [self] (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = self.textProcessor!.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: self.maxTextSequenceLength)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: maxMotionLength, discretizer: discretizer!)

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source, label: target)
            return singleBatch
        }
        print("Dataset acquired.")
        return dataset
    }

    static func getTextProcessor() -> TextProcessor {
        /// instantiate text processor
        print("instantiate text processor")
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        return TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
    }
    
    public init(datasetSize: DatasetSize, device: Device) {
        textProcessor = DatasetManager.getTextProcessor()
        discretizer = MotionDiscretizer(n_bins: 300)
        dataset = try! loadDataset(datasetSize: datasetSize, device: device)
    }
}
