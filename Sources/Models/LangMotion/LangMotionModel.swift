import TensorFlow
import Datasets


public struct LangMotionModel: Module {
    public var transformer: LangMotionTransformer
    public var mixtureModel: MotionGaussianMixtureModel

    public init(transformer: LangMotionTransformer, mixtureModel: MotionGaussianMixtureModel) {
        self.transformer = transformer
        self.mixtureModel = mixtureModel
    }

    @differentiable
    public func callAsFunction(_ input: LangMotionBatch) -> Tensor<Float> {
        return self.transformer(input)
    }

    @differentiable
    public func generate(input: LangMotionBatch) -> MixtureModelPreds {
        return self.mixtureModel(self.transformer.generator(self.callAsFunction(input)))
    }
}
