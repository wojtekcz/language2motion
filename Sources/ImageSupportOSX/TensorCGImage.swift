//
//  TensorCGImage.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 10/12/20.
//

import Foundation
import TensorFlow

// https://stackoverflow.com/questions/48312161/how-to-save-cgimage-to-data-in-swift
extension CGImage {
    public var png: Data? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
            let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
    public func toCGImage() -> CGImage {
        // convert tensor scalars to RGBA values
        // cmapRange: 1.0
        // cmap: "Spectral", vmin: -cmapRange, vmax: cmapRange

        let width = self.shape[0]
        let height = self.shape[1]
        var srgbArray = [UInt32](repeating: 0x00, count: width * height)

        // create CGImage
        for y in 0..<height {
            for x in 0..<width {
                let c: Float = Float(self[x, y].scalar!)
                let scaledC: Float = (c + 1.0)/2.0 // 0.0 ... 1.0
                srgbArray[x+width*y] = SpectralColormap.spectralColor(c: scaledC).sRGB
            }
        }
        
        // https://forums.swift.org/t/creating-a-cgimage-from-color-array/18634
        let cgImg = srgbArray.withUnsafeMutableBytes { (ptr) -> CGImage in
            let ctx = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: 4*width,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue +
                    CGImageAlphaInfo.premultipliedFirst.rawValue
            )!
            return ctx.makeImage()!
        }
        return cgImg
    }
}
