import Testing
@testable import SwipeApp

struct CaptureProgressTests {
    @Test func detectsStartupWithoutSamplesAndLossOfAudioAfterStartup() {
        var progress = CaptureProgress(now: 100)
        func stalled(_ frames: Int, _ now: Double) -> Bool { progress.isStalled(frames: frames, now: now) }
        #expect(!stalled(0, 102))
        #expect(stalled(0, 103))
        progress = CaptureProgress(now: 200)
        #expect(!stalled(48000, 201))
        #expect(!stalled(96000, 202))
        #expect(!stalled(96000, 204))
        #expect(stalled(96000, 205))
        #expect(!stalled(144000, 206))
    }
}
