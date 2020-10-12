//
//  ContentView.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import SwiftUI
import TensorFlow

struct GenOpts {
    let nSamples: Int
    let bestLogProbs: Bool
    let fixRotation: Bool
    let saveMMM: Bool
    
    let encoderSelfAttentionTemp: Float
    let decoderSourceAttentionTemp: Float
    let decoderSelfAttentionTemp: Float
    
    let maxMotionLength: Int

    let sentence: String
}

struct ContentView: View {
    @State var nSamples = "10"
    @State private var bestLogProbs = true
    @State private var fixRotation = true
    @State private var saveMmm = true
    @State var encoderSelfAttentionTemp = "1.0"
    @State var decoderSourceAttentionTemp = "1.0"
    @State var decoderSelfAttentionTemp = "1.0"
    @State var sentence = "A person is walking forwards."
    @State var maxMotionLength = "100"

    var motionGenerationManager: MotionGenerationManager
    
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
        
    var body: some View {
        VStack {
            Text("Motion generator")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Button(action: loadModel) {
                Text("Load checkpoint")
            }
            HStack {
                Text("maxMotionLength")
                TextField("maxMotionLength", text: $maxMotionLength)
            }
            HStack {
                Text("nSamples")
                TextField("nSamples", text: $nSamples)
            }
            HStack {
                Text("encoderSelfAttentionTemp")
                TextField("encoderSelfAttentionTemp", text: $encoderSelfAttentionTemp)
            }
            HStack {
                Text("decoderSourceAttentionTemp")
                TextField("decoderSourceAttentionTemp", text: $decoderSourceAttentionTemp)
            }
            HStack {
                Text("decoderSelfAttentionTemp")
                TextField("decoderSelfAttentionTemp", text: $decoderSelfAttentionTemp)
            }
            HStack {
                Text("sentence")
                TextField("sentence", text: $sentence)
            }
            Button(action: generateMotion) {
                Text("Generate motion")
            }
            VStack {
                Toggle(isOn: $bestLogProbs) {
                    Text("bestLogProbs")
                }
                Toggle(isOn: $fixRotation) {
                    Text("fix rotation")
                }
                Toggle(isOn: $saveMmm) {
                    Text("save mmm")
                }
            }
        }
    }

    func loadModel() {
        motionGenerationManager.loadModel()
    }
    
    func generateMotion() {
        print("nSamples: \(Int(nSamples) ?? 99)")
        
        let opts = GenOpts(nSamples: Int(nSamples) ?? 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMmm, encoderSelfAttentionTemp: Float(encoderSelfAttentionTemp) ?? 1.0, decoderSourceAttentionTemp: Float(decoderSourceAttentionTemp) ?? 1.0, decoderSelfAttentionTemp: Float(decoderSelfAttentionTemp) ?? 1.0, maxMotionLength: Int(maxMotionLength) ?? 10, sentence: sentence)
        
        motionGenerationManager.generateMotion(genOpts: opts)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        let motionGenerationManager = MotionGenerationManager()
        ContentView(nSamples: "10", motionGenerationManager: motionGenerationManager)
    }
}
