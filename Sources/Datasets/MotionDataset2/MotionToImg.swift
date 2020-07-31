import TensorFlow
import Foundation
import ModelSupport
import PythonKit

let plt = Python.import("matplotlib.pyplot")


extension Tensor where Scalar: NumpyScalarCompatible, Scalar: Numeric {
    public func motionToImg(url: URL, padTo: Int = 500, descr: String = "") {
        let motion = self.paddedAndCropped(to: padTo).motion
        let x = plt.subplots()
        let ax = x[1]
        // cmaps: viridis, gist_rainbow, bwr, seismic, coolwarm, hsv, plasma*, PRGn, twilight_shifted, Spectral...
        ax.imshow(motion.makeNumpyArray().T, extent: [0, padTo, 0, motion.shape[1]], cmap: "Spectral")
        ax.set_title("\(descr)")
        plt.savefig(url.path)
    }
}
