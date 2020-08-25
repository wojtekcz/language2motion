import Foundation

public struct MotionFrame {
    public static let cjpRootRotationIdx = 44
    public static let numCombinedJointPositions = cjpRootRotationIdx + 3 // 47

    public let timestep: Float
    public let rootPosition: [Float]? // 3 values
    public let rootRotation: [Float] // 3 values
    public let jointPositions: [Float] // 44 positions
    public let jointNames: [String]

    public init(timestep: Float, rootPosition: [Float]?, rootRotation: [Float], jointPositions: [Float], jointNames: [String]) {
        self.timestep = timestep
        self.rootPosition = rootPosition
        self.rootRotation = rootRotation
        self.jointPositions = jointPositions
        self.jointNames = jointNames
    }

    public func combinedJointPositions() -> [Float] {
        let combined = jointPositions[0..<44] + rootRotation
        return Array(combined)
    }

    public func jpIdx(of: String) -> Int {
        return jointNames.firstIndex(of: of)!
    }

    public func armAndHandIdxs(side: String) -> [Int] {
        // shoulder
        let Sx = self.jpIdx(of: side+"Sx")
        let Sy = self.jpIdx(of: side+"Sy")
        let Sz = self.jpIdx(of: side+"Sz")

        // elbow
        let Ex = self.jpIdx(of: side+"Ex")
        let Ez = self.jpIdx(of: side+"Ez")

        // wrist
        let Wx = self.jpIdx(of: side+"Wx")
        let Wy = self.jpIdx(of: side+"Wy")
        
        return [
            Sx, Ex, Wx,
            Sy, Wy, 
            Sz, Ez
        ]
    }

    public func legAndFootIdxs(side: String) -> [Int] {
        // hip
        let Hx = self.jpIdx(of: side+"Hx")
        let Hy = self.jpIdx(of: side+"Hy")
        let Hz = self.jpIdx(of: side+"Hz")

        // knee
        let Kx = self.jpIdx(of: side+"Kx")

        // ankle
        let Ax = self.jpIdx(of: side+"Ax")
        let Ay = self.jpIdx(of: side+"Ay")
        let Az = self.jpIdx(of: side+"Az")

        // m?
        let Mrot = self.jpIdx(of: side+"Mrot")

        // foot
        let Fx = self.jpIdx(of: side+"Fx")

        return [
            Hx, Kx, Ax, Mrot, Fx, 
            Hy, Ay, 
            Hz, Az
        ]
    }

    public func torsoHeadNeckIdxs() -> [Int] {
        // torso
        let RRx = Self.cjpRootRotationIdx + 0
        let RRy = Self.cjpRootRotationIdx + 1
        let RRz = Self.cjpRootRotationIdx + 2

        let BPx = self.jpIdx(of: "BPx")
        let BPy = self.jpIdx(of: "BPy")
        let BPz = self.jpIdx(of: "BPz")

        let BTx = self.jpIdx(of: "BTx")
        let BTy = self.jpIdx(of: "BTy")
        let BTz = self.jpIdx(of: "BTz")

        // lower neck
        let BLNx = self.jpIdx(of: "BLNx")
        let BLNy = self.jpIdx(of: "BLNy")
        let BLNz = self.jpIdx(of: "BLNz")

        // upper neck
        let BUNx = self.jpIdx(of: "BUNx")
        let BUNy = self.jpIdx(of: "BUNy")
        let BUNz = self.jpIdx(of: "BUNz")

        return [
            RRx, BPx, BTx, BLNx, BUNx, 
            RRy, BPy, BTy, BLNy, BUNy, 
            RRz, BPz, BTz, BLNz, BUNz
        ]
    }

    public static func grouppedJointPositionIdxs(jointNames: [String]) -> [Int] {
        let mf = Self(timestep: 0.0, rootPosition: [0.0], rootRotation: [0.0], jointPositions: [0.0], jointNames: jointNames)
        return mf.torsoHeadNeckIdxs() + 
        mf.armAndHandIdxs(side: "L") +
        mf.legAndFootIdxs(side: "L") +
        mf.armAndHandIdxs(side: "R") + 
        mf.legAndFootIdxs(side: "R")
    }

    /// legacy code rearranged on joint position values
    // func jp(of: String) -> Float {
    //     return self.jointPositions[jointNames.firstIndex(of: of)!]
    // }

    // func armAndHand(side: String) -> [Float] {
    //     // shoulder
    //     let Sx = self.jp(of: side+"Sx")
    //     let Sy = self.jp(of: side+"Sy")
    //     let Sz = self.jp(of: side+"Sz")

    //     // elbow
    //     var Ex = self.jp(of: side+"Ex")
    //     var Ez = self.jp(of: side+"Ez")

    //     // wrist
    //     var Wx = self.jp(of: side+"Wx")
    //     var Wy = self.jp(of: side+"Wy")
        
    //     // elbow
    //     Ex += Sx
    //     Ez += Sz
        
    //     // wrist
    //     Wx += Ex
    //     Wy += Sy
        
    //     // return [Sx, Sy, Sz, Ex, Ez, Wx, Wy]
    //     return [
    //         Sx, Ex, Wx,
    //         Sy, Wy, 
    //         Sz, Ez
    //     ]
    // }

    // func legAndFoot(side: String) -> [Float] {
    //     // hip
    //     let Hx = self.jp(of: side+"Hx")
    //     let Hy = self.jp(of: side+"Hy")
    //     let Hz = self.jp(of: side+"Hz")

    //     // knee
    //     var Kx = self.jp(of: side+"Kx")

    //     // ankle
    //     var Ax = self.jp(of: side+"Ax")
    //     var Ay = self.jp(of: side+"Ay")
    //     var Az = self.jp(of: side+"Az")

    //     // m?
    //     var Mrot = self.jp(of: side+"Mrot")

    //     // foot
    //     var Fx = self.jp(of: side+"Fx")

    //     // knee
    //     Kx += Hx
        
    //     // ankle
    //     Ax += Kx
    //     Ay += Hy
    //     Az += Hz
        
    //     // m?
    //     Mrot += Ax
        
    //     // foot
    //     Fx += Ax
        
    //     // return [Hx, Hy, Hz, Kx, Ax, Ay, Az, Mrot, Fx]
    //     return [
    //         Hx, Kx, Ax, Mrot, Fx, 
    //         Hy, Ay, 
    //         Hz, Az
    //     ]
    // }

    // func torsoHeadNeck() -> [Float] {
    //     // torso
    //     let RRx = rootRotation[0]
    //     let RRy = rootRotation[1]
    //     let RRz = rootRotation[2]

    //     var BPx = self.jp(of: "BPx")
    //     var BPy = self.jp(of: "BPy")
    //     var BPz = self.jp(of: "BPz")

    //     var BTx = self.jp(of: "BTx")
    //     var BTy = self.jp(of: "BTy")
    //     var BTz = self.jp(of: "BTz")

    //     // lower neck
    //     var BLNx = self.jp(of: "BLNx")
    //     var BLNy = self.jp(of: "BLNy")
    //     var BLNz = self.jp(of: "BLNz")

    //     // upper neck
    //     var BUNx = self.jp(of: "BUNx")
    //     var BUNy = self.jp(of: "BUNy")
    //     var BUNz = self.jp(of: "BUNz")

    //     // torso
    //     BPx += RRx
    //     BPy += RRy
    //     BPz += RRz

    //     BTx += BPx
    //     BTy += BPy
    //     BTz += BPz
        
    //     // lower neck
    //     BLNx += BTx
    //     BLNy += BTy
    //     BLNz += BTz
        
    //     // upper neck
    //     BUNx += BLNx
    //     BUNy += BLNy
    //     BUNz += BLNz
        
    //     // return [RRx, RRy, RRz, BPx, BPy, BPz, BTx, BTy, BTz, BLNx, BLNy, BLNz, BUNx, BUNy, BUNz]
    //     return [
    //         RRx, BPx, BTx, BLNx, BUNx, 
    //         RRy, BPy, BTy, BLNy, BUNy, 
    //         RRz, BPz, BTz, BLNz, BUNz
    //     ]
    // }

    // public func grouppedJointPositions() -> [Float] {
    //     return torsoHeadNeck() + 
    //     armAndHand(side: "R") + 
    //     armAndHand(side: "L") +
    //     legAndFoot(side: "R") +
    //     legAndFoot(side: "L")
    // }
}
