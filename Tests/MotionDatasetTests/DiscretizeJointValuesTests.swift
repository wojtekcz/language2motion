//
//  DiscretizeJointValuesTests.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 10/27/20.
//

import XCTest
import TensorFlow
import Datasets
import ModelSupport
import TextModels
import Foundation

class DiscretizeJointValuesTests: XCTestCase {

    var dataset: Lang2Motion? = nil
    #if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func loadData(datasetSize: DatasetSize = .full, minMotionLength: Int = 20, maxMotionLength: Int = 150) {
        /// load dataset
        print("\nLoading dataset...")

        let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

        /// instantiate text processor
        print("instantiate text processor")
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
        var discretizer = MotionDiscretizer(n_bins: 300)
        
        dataset = try! Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: 2,
            minMotionLength: minMotionLength,
            maxMotionLength: maxMotionLength,
            discretizer: &discretizer,
            trainTestSplit: 1.0,
            device: Device.defaultTFEager
        ) { (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: 40)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: 100, discretizer: discretizer)

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source, label: target)
            return singleBatch
        }
    }

    func testDiscretizing() throws {
        loadData(datasetSize: .full, minMotionLength: 10, maxMotionLength: 150)
        print("\ntest discretizing")
        let samples = dataset!.motionSamples
        
        let motions = samples.map { $0.motion }
        print("computing bins...")
        let discretizer = MotionDiscretizer(n_bins: 300, X: Tensor(concatenating: motions, alongAxis: 0))
        print("discretizing...")
        let discretizedMotions = motions.map { discretizer.transform($0) }
        print("discretizedMotions.count: \(discretizedMotions.count)")
        print("inversing...")
        let inversedMotions = discretizedMotions.map { discretizer.inverse_transform($0) }

        print("checking difference")
        let diffs = zip(inversedMotions, motions).map { abs($0 - $1).sum().scalar! }
        let totalDiff = diffs.reduce(0, +)
        print("diffs: \(totalDiff), \(totalDiff/Float(diffs.count))")
        
    }
}
