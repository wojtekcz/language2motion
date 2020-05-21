import TensorFlow

extension ResNet {
    // extra channelCount parameter
    public init(
        classCount: Int, depth: Depth, downsamplingInFirstStage: Bool = true,
        useLaterStride: Bool = true, channelCount: Int = 3
    ) {
        let inputFilters: Int
        
        if downsamplingInFirstStage {
            inputFilters = 64
            initialLayer = ConvBN(
                filterShape: (7, 7, channelCount, inputFilters), strides: (2, 2), padding: .same)
            maxPool = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .same)
        } else {
            inputFilters = 16
            initialLayer = ConvBN(filterShape: (3, 3, channelCount, inputFilters), padding: .same)
            maxPool = MaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
        }

        var lastInputFilterCount = inputFilters
        for (blockSizeIndex, blockSize) in depth.layerBlockSizes.enumerated() {
            for blockIndex in 0..<blockSize {
                let strides = ((blockSizeIndex > 0) && (blockIndex == 0)) ? (2, 2) : (1, 1)
                let filters = inputFilters * Int(pow(2.0, Double(blockSizeIndex)))
                let residualBlock = ResidualBlock(
                    inputFilters: lastInputFilterCount, filters: filters, strides: strides,
                    useLaterStride: useLaterStride, isBasic: depth.usesBasicBlocks)
                lastInputFilterCount = filters * (depth.usesBasicBlocks ? 1 : 4)
                residualBlocks.append(residualBlock)
            }
        }

        let finalFilters = inputFilters * Int(pow(2.0, Double(depth.layerBlockSizes.count - 1)))
        classifier = Dense(
            inputSize: depth.usesBasicBlocks ? finalFilters : finalFilters * 4,
            outputSize: classCount)
    }
}
