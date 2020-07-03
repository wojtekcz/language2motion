public struct Prediction {
    public let classIdx: Int
    public let className: String
    public let probability: Float

    public init(classIdx: Int, className: String, probability: Float) {
        self.classIdx = classIdx
        self.className = className
        self.probability = probability
    }
}
