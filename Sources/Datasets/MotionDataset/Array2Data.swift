import Foundation


extension Array {
    // https://stackoverflow.com/questions/49555088/swift-best-way-to-send-large-arrays-of-numbers-double-over-http

    public init?(data: Data) {
        // This check should be more complex, but here we just check if total byte count divides to one element size in bytes
        guard data.count % MemoryLayout<Element>.size == 0 else { return nil }

        let elementCount = data.count / MemoryLayout<Element>.size
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: elementCount)
        let _ = data.copyBytes(to: buffer)

        self = buffer.map({$0})
        buffer.deallocate()
    }

   // Wrapped here code above
    public var data: Data {
        return self.withUnsafeBufferPointer { pointer in
            return Data(buffer: pointer)
        }
    }
}
