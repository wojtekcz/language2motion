import Foundation
import TensorFlow


public struct MotionCatDistPreds: Differentiable {
    public var catDistProbs: Tensor<Float> // bs x motionLength x nbJoints x discreteBins
    public var stops: Tensor<Float>        // bs x motionLength x 1

    @differentiable
    public init(catDistProbs: Tensor<Float>, stops: Tensor<Float>) {
        self.catDistProbs = catDistProbs
        self.stops = stops
    }

    @differentiable
    public init(stacking preds: [MotionCatDistPreds], alongAxis axis: Int) {
        self.catDistProbs = Tensor<Float>(stacking: preds.differentiableMap{$0.catDistProbs}, alongAxis: axis)
        self.stops        = Tensor<Float>(stacking: preds.differentiableMap{$0.stops}, alongAxis: axis)
    }

    public func printPreds() {
        print("MotionCatDistPreds")
        print("  catDistProbs.shape: \(self.catDistProbs.shape)")
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

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> MotionCatDistPreds {
        let s = input.shape
        let (bs, numFrames) = (s[0], s[1])

        var catWeights1 = timeDistributed(input, catDistWeights.weight)
        catWeights1 = catWeights1.reshaped(to: [bs, numFrames, nbJoints, discreteBins])
        catWeights1 = softmax(catWeights1, alongAxis: 3)
                
        let stops = sigmoid(timeDistributed(input, linearStop.weight))
        return MotionCatDistPreds(catDistProbs: catWeights1, stops: stops)
    }
}
