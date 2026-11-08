import Testing
import AppKit
import SwipeCore
@testable import SwipeApp

struct AppLifecycleTests {
    @Test @MainActor func restoresBothMappingsWithoutOverwritingEitherPreference() throws {
        let suite = "swipe-restore-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let up = Mapping(action: .mute, shortcut: .up), down = Mapping(action: .lockScreen, shortcut: .down)
        defaults.set(try JSONEncoder().encode(up), forKey: "up")
        defaults.set(try JSONEncoder().encode(down), forKey: "down")
        let model = AppModel(defaults: defaults, workspace: NotificationCenter(), distributed: NotificationCenter())
        #expect(model.up == up)
        #expect(model.down == down)
        #expect(try JSONDecoder().decode(Mapping.self, from: defaults.data(forKey: "down")!) == down)
    }
    @Test @MainActor func controllerReconnectsAfterUnlockButNotAfterManualPause() async throws {
        let suite = "swipe-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let workspace = NotificationCenter(), distributed = NotificationCenter(), driver = SlowDriver()
        driver.release.signal(); driver.release.signal()
        let model = AppModel(defaults: defaults, microphone: Microphone(driver: driver), workspace: workspace,
                             distributed: distributed, audioPermission: { true })
        model.project = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        defer { model.stop() }
        func waitFor(_ predicate: () -> Bool) async throws {
            for _ in 0..<200 {
                if predicate() { return }
                try await Task.sleep(for: .milliseconds(25))
            }
            Issue.record("Controller did not reach expected state: \(model.status), error: \(model.error ?? "none")")
        }
        model.start(); try await waitFor { model.listening }
        #expect(model.wantsListening)
        distributed.post(name: .init("com.apple.screenIsLocked"), object: nil)
        try await waitFor { !model.active && model.status == "Waiting for unlock" }
        #expect(model.wantsListening)
        distributed.post(name: .init("com.apple.screenIsUnlocked"), object: nil)
        try await waitFor { model.listening }
        #expect(model.error == nil)
        await withCheckedContinuation { continuation in model.stop { continuation.resume() } }
        distributed.post(name: .init("com.apple.screenIsLocked"), object: nil)
        distributed.post(name: .init("com.apple.screenIsUnlocked"), object: nil)
        try await Task.sleep(for: .milliseconds(400))
        #expect(!model.active)
        #expect(!model.wantsListening)
        #expect(driver.snapshot.main == 0)
    }
}
