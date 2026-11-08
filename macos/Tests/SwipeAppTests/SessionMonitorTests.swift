import Testing
import AppKit
import SwipeCore
@testable import SwipeApp

struct SessionMonitorTests {
    @Test @MainActor func notificationReplayPausesOnLockAndResumesOnUnlock() {
        let workspace = NotificationCenter(), distributed = NotificationCenter()
        var session = ListeningSession(); session.request()
        let monitor = SessionMonitor(workspace: workspace, distributed: distributed) { blocker, present in
            session.set(blocker, present: present)
        }
        distributed.post(name: .init("com.apple.screenIsLocked"), object: nil)
        #expect(!session.shouldListen)
        workspace.post(name: NSWorkspace.willSleepNotification, object: nil)
        workspace.post(name: NSWorkspace.didWakeNotification, object: nil)
        #expect(!session.shouldListen)
        distributed.post(name: .init("com.apple.screenIsUnlocked"), object: nil)
        #expect(session.shouldListen)
        session.pause()
        distributed.post(name: .init("com.apple.screenIsUnlocked"), object: nil)
        #expect(!session.shouldListen)
        monitor.stop()
        workspace.post(name: NSWorkspace.willSleepNotification, object: nil)
        #expect(session.blockers.isEmpty)
    }
}
