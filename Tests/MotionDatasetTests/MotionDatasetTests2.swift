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
import Foundation

class MotionDatasetTests2: XCTestCase {
    
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
        
        dataset = try! Lang2Motion(
            motionDatasetURL: motionDatasetURL,
            batchSize: 2,
            minMotionLength: minMotionLength,
            maxMotionLength: maxMotionLength,
            trainTestSplit: 1.0,
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

    func testCreateSmallDatasets() throws {
        // create dataset with one sample repeated n times
        loadData(datasetSize: .multi_full, minMotionLength: 10, maxMotionLength: 50)
        
        // with smallest dataset possible
        // 2 samples
        // 2 samples + multiplied samples
        // 10 samples + multiplied samples
        // 100 samples + multiplied
        // pick short motions, < 50 frames
        
        
        let allMotionSamples = dataset!.motionSamples
        print("allMotionSamples: \(allMotionSamples.count)")

        let allSampleIDs: [Int] = Array(Set(allMotionSamples.map { $0.sampleID }))
        print("allSampleIDs: \(allSampleIDs.count)")
        
        print(( allMotionSamples.filter {$0.sampleID == 634} ).count)
        
        //   2 - small_micro
        //   2 - small_multi_micro
        //  10 - small_multi_mini
        // 100 - small_multi_midi

        func getMainMotionSamples(nSamples: Int = 1) -> [MotionSample] {
            var mainMotionSamples: [MotionSample] = []
            for _ in 0..<nSamples {
                let sampleID = allSampleIDs[Int.random(in: 0..<allSampleIDs.count)]
                let filteredSamples = allMotionSamples.filter { $0.sampleID == sampleID }
                let sample = filteredSamples[0]
                mainMotionSamples.append(sample)
            }
            return mainMotionSamples
        }

        func getMultiMotionSamples(mainMotionSamples: [MotionSample]) -> [MotionSample] {
            var multiMotionSamples: [MotionSample] = []
            for mainSample in mainMotionSamples {
                let samples = allMotionSamples.filter { $0.sampleID == mainSample.sampleID && $0.annotations[0] == mainSample.annotations[0] }
                multiMotionSamples.append(contentsOf: samples)
            }
            return multiMotionSamples
        }

        func writeDataset(datasetSize: DatasetSize, motionSamples: [MotionSample]) {
            let serializedDatasetURL = dataURL.appendingPathComponent("motion_dataset_v3.10Hz.\(datasetSize.rawValue)plist")
            let outputDataset = MotionDataset(datasetFolderURL: dataURL, motionSamples: motionSamples)
            outputDataset.write(to: serializedDatasetURL)
        }
        var mainMotionSamples: [MotionSample] = []
        var multiMotionSamples: [MotionSample] = []

        // small_micro
        mainMotionSamples = getMainMotionSamples(nSamples: 2)
        print("small_micro.unique: \(mainMotionSamples.count)")
        writeDataset(datasetSize: .small_micro, motionSamples: mainMotionSamples)
        
        // small_multi_micro
        multiMotionSamples = getMultiMotionSamples(mainMotionSamples: mainMotionSamples)
        print("small_multi_micro.multi: \(multiMotionSamples.count)")
        writeDataset(datasetSize: .small_multi_micro, motionSamples: multiMotionSamples)
        
        // small_multi_mini
        mainMotionSamples = getMainMotionSamples(nSamples: 10)
        print("small_multi_mini.unique: \(mainMotionSamples.count)")
        multiMotionSamples = getMultiMotionSamples(mainMotionSamples: mainMotionSamples)
        print("small_multi_mini.multi: \(multiMotionSamples.count)")
        writeDataset(datasetSize: .small_multi_mini, motionSamples: multiMotionSamples)

        // small_multi_midi
        mainMotionSamples = getMainMotionSamples(nSamples: 100)
        print("small_multi_midi.unique: \(mainMotionSamples.count)")
        multiMotionSamples = getMultiMotionSamples(mainMotionSamples: mainMotionSamples)
        print("small_multi_midi.multi: \(multiMotionSamples.count)")
        writeDataset(datasetSize: .small_multi_midi, motionSamples: multiMotionSamples)
    }
}
