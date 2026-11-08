/// Check delivered frames, rather than an audio API's optimistic "running" flag.
struct CaptureProgress {
    private var lastFrames = 0
    private var lastProgress: Double
    init(now: Double) { lastProgress = now }
    mutating func isStalled(frames: Int, now: Double) -> Bool {
        if frames != lastFrames { lastFrames = frames; lastProgress = now }
        return now - lastProgress >= 3
    }
}
