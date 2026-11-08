import AppKit
import SwipeCore

/// Observe screen lock separately from user switching and display/system sleep.
/// Lock notifications follow the macOS convention also used by Hammerspoon's watcher.
final class SessionMonitor {
    private var observers: [(NotificationCenter, NSObjectProtocol)] = []
    init(workspace: NotificationCenter = NSWorkspace.shared.notificationCenter,
         distributed: NotificationCenter = DistributedNotificationCenter.default(),
         changed: @escaping (ListeningSession.Blocker, Bool) -> Void) {
        func observe(_ center: NotificationCenter, _ name: Notification.Name, _ blocker: ListeningSession.Blocker, _ present: Bool) {
            let token = center.addObserver(forName: name, object: nil, queue: .main) { _ in changed(blocker, present) }
            observers.append((center, token))
        }
        observe(distributed, Notification.Name("com.apple.screenIsLocked"), .locked, true)
        observe(distributed, Notification.Name("com.apple.screenIsUnlocked"), .locked, false)
        observe(workspace, NSWorkspace.willSleepNotification, .sleeping, true)
        observe(workspace, NSWorkspace.didWakeNotification, .sleeping, false)
        observe(workspace, NSWorkspace.screensDidSleepNotification, .displayOff, true)
        observe(workspace, NSWorkspace.screensDidWakeNotification, .displayOff, false)
        observe(workspace, NSWorkspace.sessionDidResignActiveNotification, .inactive, true)
        observe(workspace, NSWorkspace.sessionDidBecomeActiveNotification, .inactive, false)
    }
    func stop() { for (center, token) in observers { center.removeObserver(token) }; observers.removeAll() }
    deinit { stop() }
}
