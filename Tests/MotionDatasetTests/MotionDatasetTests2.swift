//
//  MotionDatasetTests2.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 10/13/20.
//

import XCTest
import TensorFlow
import Datasets
import ModelSupport
import TextModels

class MotionDatasetTests2: XCTestCase {
    
    var dataset: Lang2Motion? = nil
    #if os(macOS)
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
    #else
    let dataURL = URL(fileURLWithPath: "/notebooks/language2motion.gt/data/")
    #endif

    func loadData() {
        /// load dataset
        print("\nLoading dataset...")

        let datasetSize: DatasetSize = .full

        let motionDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")

        /// instantiate text processor
        print("instantiate text processor")
        let vocabularyURL = dataURL.appendingPathComponent("vocab.txt")
        let vocabulary: Vocabulary = try! Vocabulary(fromFile: vocabularyURL)
        let tokenizer: Tokenizer = BERTTokenizer(vocabulary: vocabulary, caseSensitive: false, unknownToken: "[UNK]", maxTokenLength: nil)
        let textProcessor = TextProcessor(vocabulary: vocabulary, tokenizer: tokenizer)
        
        dataset = try! Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: 100,
            minMotionLength: 20,
            maxMotionLength: 150,
            trainTestSplit: 0.9,
            device: Device.defaultTFEager
        ) { (motionSample: MotionSample) -> LangMotionBatch in
            let sentence = textProcessor.preprocess(sentence: motionSample.annotations[0], maxTextSequenceLength: 40)
            let (motionPart, target) = LangMotionBatch.preprocessTargetMotion(sampleID: motionSample.sampleID, motion: motionSample.motion, maxMotionLength: 100)

            let source = LangMotionBatch.Source(sentence: sentence, motionPart: motionPart)
            let singleBatch = LangMotionBatch(data: source, label: target)
            return singleBatch
        }
    }

    func testCreateSameDataset() throws {
        // create dataset with one sample repeated n times
        loadData()
        
        var motionSamples: [MotionSample] = []
        
        let sample = dataset!.motionSampleDict[1]!
        
        // 10000 - same.midi
        // 1000 - same.mini
        // 1000 - same.micro

        for _ in 0..<100 {
            motionSamples.append(sample)
        }
        let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.same.micro.plist")
        let outputDataset = MotionDataset(datasetFolderURL: dataURL, motionSamples: motionSamples)
        outputDataset.write(to: serializedDatasetURL)
    }
}
