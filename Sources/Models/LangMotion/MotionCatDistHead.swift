import Foundation
import TensorFlow
import Datasets
import PythonKit


public struct MotionCatDistPreds: Differentiable {
    public var catDistLogits: Tensor<Float> // bs x motionLength x nbJoints x discreteBins
    public var stops: Tensor<Float>        // bs x motionLength x 1

    @differentiable
    public init(catDistLogits: Tensor<Float>, stops: Tensor<Float>) {
        self.catDistLogits = catDistLogits
        self.stops = stops
    }

    @differentiable
    public init(stacking preds: [MotionCatDistPreds], alongAxis axis: Int) {
        self.catDistLogits = Tensor<Float>(stacking: preds.differentiableMap{$0.catDistLogits}, alongAxis: axis)
        self.stops         = Tensor<Float>(stacking: preds.differentiableMap{$0.stops}, alongAxis: axis)
    }

    public func printPreds() {
        print("MotionCatDistPreds")
        print("  catDistLogits.shape: \(self.catDistLogits.shape)")
        print("  stops.shape: \(self.stops.shape)")
    }
}


public struct MotionCatDistHead: Module {

    @noDerivative public var inputSize: Int
    @noDerivative public var nbJoints: Int
    @noDerivative public var discreteBins: Int

    public var catDistWeights: Dense<Float>
    public var linearStop: Dense<Float>

    public init(inputSize: Int, nbJoints: Int, discreteBins: Int) {
        self.inputSize = inputSize
        self.nbJoints = nbJoints
        self.discreteBins = discreteBins

        catDistWeights = Dense<Float>(inputSize: inputSize, outputSize: nbJoints*discreteBins)

        // and stop bit
        linearStop = Dense<Float>(inputSize: inputSize, outputSize: 1)
    }

    public init(
        inputSize: Int, nbJoints: Int, discreteBins: Int,
        catDistWeights: Dense<Float>, linearStop: Dense<Float>
    ) {
        self.inputSize = inputSize
        self.nbJoints = nbJoints
        self.discreteBins = discreteBins
        self.catDistWeights = catDistWeights
        self.linearStop = linearStop
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> MotionCatDistPreds {
        let s = input.shape
        let (bs, numFrames) = (s[0], s[1])

        var catLogits = timeDistributed(input, catDistWeights.weight)
        catLogits = catLogits.reshaped(to: [bs, numFrames, nbJoints, discreteBins])
        //catLogits = softmax(catLogits, alongAxis: 3)
                
        let stops = sigmoid(timeDistributed(input, linearStop.weight))
        return MotionCatDistPreds(catDistLogits: catLogits, stops: stops)
    }
}

//let np = Python.import("numpy")
public func sampleCatDistMotion(catDistLogits: Tensor<Float>) -> Tensor<Int32> {
    var samples: [Int32] = []
    let sh = catDistLogits.shape
    let (bs, nFrames, nbJoints) = (sh[0], sh[1], sh[2])
    for s in 0..<bs {
        for f in 0..<nFrames {
            for j in 0..<nbJoints {
                let pvals = softmax(catDistLogits[s, f, j]).scalars.map { Double($0)}
                // TODO: try to make sampling faster with a tensorflow call
                let sample: Int32 = Int32(np.argmax(randomNumber(probabilities: pvals)))!
                //let sample: Int32 = Int32(np.argmax(np.random.multinomial(1, pvals)))!
                samples.append(sample)
            }
        }
    }
    let samplesTensor = Tensor<Int32>(shape: [bs, nFrames, nbJoints], scalars: samples)
    return samplesTensor
}

public struct CDLossArgs {
    public let device: Device

    public init(device: Device) {
        self.device = device
     }
}

@differentiable(wrt: y_pred)
public func _categoryDistributionSurrogateLoss(y_true: LangMotionBatch.Target, y_pred: MotionCatDistPreds, args: CDLossArgs) -> Tensor<Float> {
    
    // use categorical cross-entropy loss (over discretized joint positions)
    let labels = y_true.discreteMotion.reshaped(to: [-1])
    let sh = y_true.discreteMotion.shape
    let resultSize =  sh[0] * sh[1] * sh[2]
    let logits = y_pred.catDistLogits.reshaped(to: [resultSize, -1])

    @differentiable
    func _none(t: Tensor<Float>) -> Tensor<Float> { t }
    var catDistLoss = softmaxCrossEntropy(logits: logits, labels: labels, reduction: _none).reshaped(like: y_true.discreteMotion)

    let stops = y_pred.stops.squeezingShape(at: 2)
    
    // mask mixture loss with stops
    let zeroTensor = Tensor<Float>(repeating: 0.0, shape: catDistLoss.shape, on: args.device)
    let true_stopsBcasted = y_true.stops.expandingShape(at: 2).broadcasted(like: zeroTensor)
    catDistLoss = catDistLoss.replacing(with: zeroTensor, where: true_stopsBcasted .== Tensor<Float>(1.0, on: args.device))

    // computer stop loss
    let bernoulli_pdf = y_true.stops * stops + (Float(1.0) - y_true.stops) * (Float(1.0) - stops)

    let loss = catDistLoss.mean(alongAxes: 2).squeezingShape(at: 2) + bernoulli_pdf
    return loss
}

@differentiable(wrt: y_pred)
public func categoryDistributionSurrogateLoss(y_pred: MotionCatDistPreds, y_true: LangMotionBatch.Target, args: CDLossArgs) -> Tensor<Float> {
    let losses = _categoryDistributionSurrogateLoss(y_true: y_true, y_pred: y_pred, args: args)
    
    if losses.isNaN.any() {
        print("\nnans in losses: \(Tensor<Float>(losses.isNaN).sum())")
    }
    
    // masking
    let segmentIDs = Tensor<Float>(y_true.segmentIDs)
    let paddingTensor = Tensor(Float(LangMotionBatch.MotionSegment.padding.rawValue), on: args.device)
    let onesTensor = Tensor(repeating: Float(1.0), shape: segmentIDs.shape, on: args.device)
    let lossesMask = segmentIDs.replacing(with: onesTensor, where: segmentIDs .!= paddingTensor)
    return (losses * lossesMask).sum()/lossesMask.sum()
}
