//
//  ContentView.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import SwiftUI
import TensorFlow

struct ContentView: View {
    @State var nSamples: String
    
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
                TextField("maxMotionLength", text: $nSamples)
            }
            HStack {
                Text("nSamples")
                TextField("samples", text: $nSamples)
            }
            HStack {
                Text("encoderSelfAttentionTemp")
                TextField("encoderSelfAttentionTemp", text: $nSamples)
            }
            HStack {
                Text("decoderSourceAttentionTemp")
                TextField("decoderSourceAttentionTemp", text: $nSamples)
            }
            HStack {
                Text("decoderSelfAttentionTemp")
                TextField("decoderSelfAttentionTemp", text: $nSamples)
            }
            HStack {
                Text("sentence")
                TextField("sentence", text: $nSamples)
            }
            Button(action: generateMotion) {
                Text("Generate motion")
            }
        }
    }

    func loadModel() {
        motionGenerationManager.loadModel()
    }
    
    func generateMotion() {
        print("nSamples: \(nSamples)")
        motionGenerationManager.generateMotion(nSamples: nSamples)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        let motionGenerationManager = MotionGenerationManager()
        ContentView(nSamples: "10", motionGenerationManager: motionGenerationManager)
    }
}
