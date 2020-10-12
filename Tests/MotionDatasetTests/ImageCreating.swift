//
//  ImageCreating.swift
//  MotionDatasetTests
//
//  Created by Wojciech Czarnowski on 10/12/20.
//

import XCTest
import TensorFlow
import Foundation
import ModelSupport

class ImageCreating: XCTestCase {

    func testSpectralColormap() throws {
        // load tensor
        let url = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/runs/Lang2motion/full_7x/run_75/generated_motions_app/epoch_100_motion_1.tensor")

        // convert tensor scalars to RGBA values
        // cmapRange: 1.0
        // cmap: "Spectral", vmin: -cmapRange, vmax: cmapRange
        let width = 500
        let height = 100
        var srgbArray = [UInt32](repeating: 0xFF204080, count: width * height)

        // create CGImage
        for y in 0..<height {
            for x in 0..<width {
                let c: Float = -1 + 2*Float(x)/Float(width) // -1.0 ... 1.0
                let scaledC: Float = (c + 1.0)/2.0 // 0.0 ... 1.0
                srgbArray[x+width*y] = SpectralColormap.spectralColor(c: scaledC).sRGB
            }
        }
        
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
        
        print(cgImg)
        // save image to PNG
        try! cgImg.png!.write(to: url.appendingPathExtension("png"))
    }

    func testImageCreating() throws {
        // load tensor
        let url = URL(fileURLWithPath: "/Users/wcz/Beanflows/All_Beans/swift4tf/language2motion.gt/data/runs/Lang2motion/full_7x/run_75/generated_motions_app/epoch_100_motion_1.tensor")
        let decoder = PropertyListDecoder()
        let data = try! Data(contentsOf: url)
        let decoded = try decoder.decode(Tensor<Float>.self, from: data)
        print(decoded.shape)

        // convert tensor scalars to RGBA values
        // cmapRange: 1.0
        // cmap: "Spectral", vmin: -cmapRange, vmax: cmapRange
        let cgImg = decoded.toCGImage()
        print(cgImg)
        // save image to PNG
        try! cgImg.png!.write(to: url.appendingPathExtension("png"))
    }
}
