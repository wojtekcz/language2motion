import TensorFlow


public struct MixtureModelPreds: Differentiable {
    public var mixtureMeans: Tensor<Float>   // bs x motionLength x nbJoints * nbMixtures
    public var mixtureVars: Tensor<Float>    // bs x motionLength x nbJoints * nbMixtures
    public var mixtureWeights: Tensor<Float> // bs x motionLength x nbMixtures
    public var stops: Tensor<Float>          // bs x motionLength x 1

    @differentiable
    public init(mixtureMeans: Tensor<Float>, mixtureVars: Tensor<Float>, mixtureWeights: Tensor<Float>, stops: Tensor<Float>) {
        self.mixtureMeans = mixtureMeans
        self.mixtureVars = mixtureVars
        self.mixtureWeights = mixtureWeights
        self.stops = stops
    }

    @differentiable
    public init(stacking preds: [MixtureModelPreds], alongAxis axis: Int) {
        self.mixtureMeans   = Tensor<Float>(stacking: preds.differentiableMap{$0.mixtureMeans}, alongAxis: axis)
        self.mixtureVars    = Tensor<Float>(stacking: preds.differentiableMap{$0.mixtureVars}, alongAxis: axis)
        self.mixtureWeights = Tensor<Float>(stacking: preds.differentiableMap{$0.mixtureWeights}, alongAxis: axis)
        self.stops          = Tensor<Float>(stacking: preds.differentiableMap{$0.stops}, alongAxis: axis)
    }

    public func printPreds() {
        print("preds")
        print("  mixtureMeans.shape: \(self.mixtureMeans.shape)")
        print("  mixtureVars.shape: \(self.mixtureVars.shape)")
        print("  mixtureWeights.shape: \(self.mixtureWeights.shape)")
        print("  stops.shape: \(self.stops.shape)")
    }
}
