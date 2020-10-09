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
            Text("Hello, World!")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Button(action: loadDataset) {
                Text("Load dataset")
            }
            Button(action: loadModel) {
                Text("Load model")
            }
            Button(action: generateMotion) {
                Text("Generate motion")
            }
            HStack {
                Text("nSamples")
                TextField("samples", text: $nSamples)
            }
        }
    }
    
    func loadDataset() {
        print("not needed anymore")
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
