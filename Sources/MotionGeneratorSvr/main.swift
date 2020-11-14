import Foundation
import MotionGenerator

print("MotionGenerator")

let motionGenerationManager = MotionGenerationManager()
motionGenerationManager.loadDataset()
motionGenerationManager.loadModel()

func generateMotion() {
    let bestLogProbs = true
    let fixRotation = true
    let saveMMM = true
    let maxMotionLength = 50
    let sentence = "A person is walking forwards."

    let opts = GenOpts(nSamples: 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMMM, encoderSelfAttentionTemp: 1.0, decoderSourceAttentionTemp: 1.0, decoderSelfAttentionTemp: 1.0, maxMotionLength: maxMotionLength, sentence: sentence)
    
    let _ = motionGenerationManager.generateMotion(genOpts: opts)
//    motionCGImage = tensor.toCGImage()
}

generateMotion()
