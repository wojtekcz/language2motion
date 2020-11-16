import TensorFlow


extension CheckpointReader {
    public func readTensor<Scalar: TensorFlowScalar>(
        name: String
    ) -> Tensor<Scalar> {
        return Tensor<Scalar>(loadTensor(named: name))
    }
}
