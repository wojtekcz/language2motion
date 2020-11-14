import Foundation
import MotionGenerator

print("MotionGenerator")

let mgMgr = MotionGenerationManager()
mgMgr.loadDataset(datasetSize: .micro, maxSamples: nil, maxMotionLength: 75)
mgMgr.loadModel(logdir: "runs/Lang2motion/", runName: "run_176", modelName: "model.e3")

func generateMotion() {
    let bestLogProbs = true
    let fixRotation = true
    let saveMMM = true
    let maxMotionLength = 50
    let sentence = "A person is walking forwards."

    let opts = GenOpts(nSamples: 10, bestLogProbs: bestLogProbs, fixRotation: fixRotation, saveMMM: saveMMM, encoderSelfAttentionTemp: 1.0, decoderSourceAttentionTemp: 1.0, decoderSelfAttentionTemp: 1.0, maxMotionLength: maxMotionLength, sentence: sentence)
    
    let _ = mgMgr.generateMotion(genOpts: opts, prefix: "temp_motion")
}

generateMotion()
