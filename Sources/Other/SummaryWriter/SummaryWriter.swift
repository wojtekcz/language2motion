import Foundation
import TensorFlow


public class SummaryWriter {
    public let writerHandle: ResourceHandle
    
    public init(logdir: URL, flushMillis: Int) {
        writerHandle = _Raw.summaryWriter(sharedName: logdir.path, container: "summary_writer_container")
        let flushMillisTensor = Tensor(Int32(flushMillis))
        _Raw.createSummaryFileWriter(
            writer: writerHandle,
            logdir: StringTensor(logdir.path),
            maxQueue: Tensor(1),
            flushMillis: flushMillisTensor, 
            filenameSuffix: StringTensor("")
        )
    }

    public func writeScalarSummary(tag: String, step: Int, value: Float) {
        let stepTensor = Tensor(Int64(step))
        let tagTensor = StringTensor(tag)
        let valueTensor = Tensor<Float>(value)
        _Raw.writeScalarSummary(writer: writerHandle, step: stepTensor, tag: tagTensor, value: valueTensor)
    }

    public func flush() {
        _Raw.flushSummaryWriter(writer: writerHandle)
    }
}
