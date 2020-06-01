import Foundation

public struct MotionFrame {
    public let timestamp: Float
    public let jointPositions: [Float]
    public let jointNames: [String]

    func jp(of: String) -> Float {
        return self.jointPositions[jointNames.firstIndex(of: of)!]
    }

    func armAndHand(side: String) -> [Float] {
        // shoulder
        let Sx = self.jp(of: side+"Sx")
        let Sy = self.jp(of: side+"Sy")
        let Sz = self.jp(of: side+"Sz")

        // elbow
        var Ex = self.jp(of: side+"Ex")
        var Ez = self.jp(of: side+"Ez")

        // wrist
        var Wx = self.jp(of: side+"Wx")
        var Wy = self.jp(of: side+"Wy")
        
        // elbow
        Ex += Sx
        Ez += Sz
        
        // wrist
        Wx += Ex
        Wy += Sy
        
        // return [Sx, Sy, Sz, Ex, Ez, Wx, Wy]
        return [
            Sx, Ex, Wx,
            Sy, Wy, 
            Sz, Ez
        ]
    }

    func legAndFoot(side: String) -> [Float] {
        // hip
        let Hx = self.jp(of: side+"Hx")
        let Hy = self.jp(of: side+"Hy")
        let Hz = self.jp(of: side+"Hz")

        // knee
        var Kx = self.jp(of: side+"Kx")

        // ankle
        var Ax = self.jp(of: side+"Ax")
        var Ay = self.jp(of: side+"Ay")
        var Az = self.jp(of: side+"Az")

        // m?
        var Mrot = self.jp(of: side+"Mrot")

        // foot
        var Fx = self.jp(of: side+"Fx")

        // knee
        Kx += Hx
        
        // ankle
        Ax += Kx
        Ay += Hy
        Az += Hz
        
        // m?
        Mrot += Ax
        
        // foot
        Fx += Ax
        
        // return [Hx, Hy, Hz, Kx, Ax, Ay, Az, Mrot, Fx]
        return [
            Hx, Kx, Ax, Mrot, Fx, 
            Hy, Ay, 
            Hz, Az
        ]
    }

    func torsoHeadNeck() -> [Float] {
        // torso
        let BPx = self.jp(of: "BPx")
        let BPy = self.jp(of: "BPy")
        let BPz = self.jp(of: "BPz")

        var BTx = self.jp(of: "BTx")
        var BTy = self.jp(of: "BTy")
        var BTz = self.jp(of: "BTz")

        // lower neck
        var BLNx = self.jp(of: "BLNx")
        var BLNy = self.jp(of: "BLNy")
        var BLNz = self.jp(of: "BLNz")

        // upper neck
        var BUNx = self.jp(of: "BUNx")
        var BUNy = self.jp(of: "BUNy")
        var BUNz = self.jp(of: "BUNz")

        // torso
        BTx += BPx
        BTy += BPy
        BTz += BPz
        
        // lower neck
        BLNx += BTx
        BLNy += BTy
        BLNz += BTz
        
        // upper neck
        BUNx += BLNx
        BUNy += BLNy
        BUNz += BLNz
        
        // return [BPx, BPy, BPz, BTx, BTy, BTz, BLNx, BLNy, BLNz, BUNx, BUNy, BUNz]
        return [
            BPx, BTx, BLNx, BUNx, 
            BPy, BTy, BLNy, BUNy, 
            BPz, BTz, BLNz, BUNz
        ]
    }

    func grouppedJointPositions() -> [Float] {
        return torsoHeadNeck() + 
        armAndHand(side: "R") + 
        armAndHand(side: "L") +
        legAndFoot(side: "R") +
        legAndFoot(side: "L") +
        [self.jointPositions[44]]
    }
}
