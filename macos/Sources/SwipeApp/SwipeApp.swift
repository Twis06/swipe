import AppKit
import SwiftUI

@main struct SwipeMain {
    @MainActor static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = AppDelegate(); app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}
@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = AppModel()
    private var statusItem: NSStatusItem!
    private var window: NSWindow!
    private var startItem: NSMenuItem!
    private var statusLabel: NSMenuItem!
    private var menuState: String?
    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "arrow.up.arrow.down", accessibilityDescription: "Swipe")
        let menu = NSMenu()
        statusLabel = NSMenuItem(title: "Swipe · Paused", action: nil, keyEquivalent: ""); menu.addItem(statusLabel)
        startItem = NSMenuItem(title: "Start listening", action: #selector(toggle), keyEquivalent: ""); startItem.target = self; menu.addItem(startItem)
        menu.addItem(.separator())
        let configure = NSMenuItem(title: "Configure actions…", action: #selector(showWindow), keyEquivalent: ","); configure.target = self; menu.addItem(configure)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Swipe", action: #selector(quitApp), keyEquivalent: "q"); quit.target = self; menu.addItem(quit)
        statusItem.menu = menu
        // An accessory app still needs a main menu for standard keyboard equivalents.
        let mainMenu = NSMenu(), appEntry = NSMenuItem(), appMenu = NSMenu()
        let settingsCommand = NSMenuItem(title: "Configure actions…", action: #selector(showWindow), keyEquivalent: ",")
        settingsCommand.target = self; appMenu.addItem(settingsCommand)
        appMenu.addItem(NSMenuItem(title: "Close window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w"))
        appMenu.addItem(.separator())
        let quitCommand = NSMenuItem(title: "Quit Swipe", action: #selector(quitApp), keyEquivalent: "q")
        quitCommand.target = self; appMenu.addItem(quitCommand)
        appEntry.submenu = appMenu; mainMenu.addItem(appEntry); NSApp.mainMenu = mainMenu
        let controller = NSHostingController(rootView: ContentView(model: model))
        controller.sizingOptions = [.preferredContentSize]
        window = NSWindow(contentViewController: controller)
        window.title = "Swipe"; window.styleMask = [.titled, .closable, .miniaturizable]
        window.isReleasedWhenClosed = false; window.center(); window.setFrameAutosaveName("SwipeSettings")
        model.stateChanged = { [weak self] in
            guard let self else { return }
            let snapshot = "\(self.model.wantsListening)|\(self.model.status)"
            guard self.menuState != snapshot else { return }; self.menuState = snapshot
            self.startItem.title = self.model.wantsListening ? "Pause listening" : "Start listening"
            self.statusLabel.title = "Swipe · \(self.model.status)"
            self.statusItem.button?.appearsDisabled = !self.model.active
        }
        model.stateChanged?(); showWindow()
    }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { showWindow(); return true }
    @objc func showWindow() { window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true) }
    @objc func toggle() { if model.wantsListening { model.stop() } else { model.start(); if model.error != nil { showWindow() } } }
    @objc func quitApp() { NSApp.terminate(nil) }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        model.stop { sender.reply(toApplicationShouldTerminate: true) }
        return .terminateLater
    }
}
