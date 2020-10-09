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
    
    var body: some View {
        VStack {
            Text("Hello, World!")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                Button(action: buttonAction) {
                    Text("Press me!")
                }
            HStack {
                Text("nSamples")
                TextField("samples", text: $nSamples)
            }
        }
    }
    
    func buttonAction() {
        print("Ala ma kota")
        print("nSamples: \(nSamples)")
        let t1 = Tensor<Float>([1.0, 2.0, 3.0])
        print(t1 .* t1)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(nSamples: "10")
    }
}
