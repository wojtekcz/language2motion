//
//  ContentView.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import SwiftUI
import TensorFlow
import ModelSupport
import AppKit

public struct GenOpts {
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
    @State private var nSamples = "1"
    @State private var bestLogProbs = true
    @State private var fixRotation = true
    @State private var saveMmm = true
    @State var encoderSelfAttentionTemp = "1.0"
    @State var decoderSourceAttentionTemp = "1.0"
    @State var decoderSelfAttentionTemp = "100000.0"
    @State var sentence = "A person is walking forwards."
    @State var maxMotionLength = "50"
//    @State var motionImageName = "motion_image"
    @State var motionCGImage: CGImage? = NSImage(named: "motion_image")?.cgImage(forProposedRect: nil, context: nil, hints: nil)

    var motionGenerationManager: MotionGenerationManager
    
    let dataURL = URL(fileURLWithPath: "/Volumes/Macintosh HD/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/")
        
    var body: some View {
        VStack {
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
                .padding(.horizontal)
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
//                HStack {
//                    Text("motion image")
//                    TextField("motion image", text: $motionImageName)
//                }
                Button(action: generateMotion) {
                    Text("Generate motion")
                }
            }
            VStack {
                Image(decorative:motionCGImage!, scale: 0.5, orientation: .up).frame(width: 300.0, height: 100.0)
            }
            .frame(width: 300.0, height: 100.0)
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
        .padding(/*@START_MENU_TOKEN@*/.all, 2.0/*@END_MENU_TOKEN@*/)
    }

    func loadModel() {
        motionGenerationManager.loadModel()
    }
    
    func generateMotion() {
        
//        let nsImage = NSImage(named: "foo")
        //let cgImage2 = NSImage(named: "foo")?.cgImage(forProposedRect: nil, context: nil, hints: nil)
        
        
//        print("nSamples: \(Int(nSamples) ?? 99)")
        
        let opts = GenOpts(nSamples: Int(nSamples) ?? 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMmm, encoderSelfAttentionTemp: Float(encoderSelfAttentionTemp) ?? 1.0, decoderSourceAttentionTemp: Float(decoderSourceAttentionTemp) ?? 1.0, decoderSelfAttentionTemp: Float(decoderSelfAttentionTemp) ?? 1.0, maxMotionLength: Int(maxMotionLength) ?? 10, sentence: sentence)
        
        let tensor = motionGenerationManager.generateMotion(genOpts: opts)
        motionCGImage = tensor.toCGImage()
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        let motionGenerationManager = MotionGenerationManager()
        ContentView(motionGenerationManager: motionGenerationManager)
    }
}
