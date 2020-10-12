//
//  AppDelegate.swift
//  test2
//
//  Created by Wojciech Czarnowski on 10/9/20.
//

import Cocoa
import SwiftUI

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow!


    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Create the SwiftUI view that provides the window contents.
        let motionGenerationManager = MotionGenerationManager()
        motionGenerationManager.loadDataset()
        motionGenerationManager.loadModel()
        let contentView = ContentView(motionGenerationManager: motionGenerationManager)

        // Create the window and set the content view.
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 480, height: 300),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.center()
        window.setFrameAutosaveName("Main Window")
        window.contentView = NSHostingView(rootView: contentView)
        window.makeKeyAndOrderFront(nil)
        
//        let t1 = Tensor<Float>([1.0, 2.0, 3.0])
//        print(t1 .* t1)
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }


}
