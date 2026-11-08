import Foundation
import Testing
import SwipeCore
@testable import SwipeApp

final class SlowDriver: MicrophoneDriver, @unchecked Sendable {
    var onCandidate: ((Candidate, Int) -> Void)?
    var onLevel: ((Float, Bool) -> Void)?
    var onFailure: ((String) -> Void)?
    let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var mainThreadCalls = 0
    private var stops = 0
    var snapshot: (main: Int, stops: Int) { lock.lock(); defer { lock.unlock() }; return (mainThreadCalls, stops) }
    func start() throws -> String {
        lock.lock(); if Thread.isMainThread { mainThreadCalls += 1 }; lock.unlock()
        entered.signal()
        guard release.wait(timeout: .now() + 3) == .success else { throw SwipeError("Test driver release timed out") }
        onLevel?(0.2, false)
        return "Test microphone"
    }
    func stop() {
        lock.lock(); if Thread.isMainThread { mainThreadCalls += 1 }; stops += 1; lock.unlock()
        Thread.sleep(forTimeInterval: 0.15) // models synchronous AudioQueueDispose
        onFailure?("Late callback from a disposed device")
    }
    func accept(_ id: Int) {}
}

struct MicrophoneControlTests {
    @Test @MainActor func blockedStartupDoesNotBlockMainQueueAndPauseCancelsLateCallbacks() async {
        let driver = SlowDriver()
        let control = Microphone(driver: driver)
        var completed = false, received = false
        control.start(candidate: { _, _ in }, level: { _, _ in received = true }, failure: { _ in received = true }) { _ in completed = true }
        let entered = await withCheckedContinuation { continuation in
            DispatchQueue.global().async { continuation.resume(returning: driver.entered.wait(timeout: .now() + 2) == .success) }
        }
        #expect(entered)
        // This main-queue heartbeat must run while the real driver seam is blocked.
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        #expect(!completed)
        let (stopped, finished) = AsyncStream<Void>.makeStream()
        control.stop { finished.yield(); finished.finish() }
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        driver.release.signal()
        for await _ in stopped {}
        #expect(!completed)
        #expect(!received)
        #expect(driver.snapshot.main == 0)
        #expect(driver.snapshot.stops >= 1)
    }
}
