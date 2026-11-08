import Foundation

public struct Candidate {
    public let id: Int, phase: String, pcm: [Float], startFrame: Int
    public let onsetFrame: Int, emitFrame: Int, sourceStartFrame: Int, sourceEndFrame: Int
    public let sourceSamples: Int, destinationOffset: Int
}

/// Sample-for-sample port of web/cascade-preview-worklet.js. Feed 128-frame blocks.
/// Confined to the audio processing queue; acceptance is dispatched to that queue too.
public final class GestureCapture {
    public let rate: Int
    private var ring: [Float], cursor = 0, frames = 0, noise = 0.0001
    private var serial = 0, id: Int?, onset = 0, cooldown = 0, quietFrames = 0
    private var offThreshold = 0.0015, capture: [Float] = [], sourceStart = 0, earlySent = false
    public private(set) var stopped = false
    public var onCandidate: ((Candidate) -> Void)?
    public init(rate: Int) { self.rate = rate; ring = .init(repeating: 0, count: rate * 2) }
    private func rounded(_ seconds: Double) -> Int { Int((Double(rate) * seconds).rounded()) }
    private func clear() { id = nil; capture.removeAll(keepingCapacity: true); earlySent = false; quietFrames = 0 }
    public func stop() { stopped = true; clear() }
    public func accept(_ acceptedID: Int) {
        if id == acceptedID && earlySent { clear(); cooldown = frames + rounded(0.4) }
    }
    public func process(_ channel: [Float]) {
        guard !stopped, !channel.isEmpty else { return }
        var power = 0.0
        for sample in channel { ring[cursor] = sample; cursor = (cursor + 1) % ring.count; power += Double(sample) * Double(sample) }
        frames += channel.count
        let rms = sqrt(power / Double(channel.count)), warming = frames < rate
        if warming { noise = 0.98 * noise + 0.02 * rms }
        let threshold = max(0.003, noise * 6)
        var triggered = false
        if !warming && id == nil && frames >= cooldown {
            if rms > threshold {
                serial += 1; id = serial; onset = frames; quietFrames = 0; offThreshold = threshold * 0.5; earlySent = false
                let preroll = rounded(0.1); sourceStart = frames - preroll
                capture = (0..<preroll).map { ring[(cursor - preroll + $0 + ring.count) % ring.count] }
                triggered = true
            } else { noise = 0.995 * noise + 0.005 * rms }
        }
        guard let activeID = id else { return }
        if !triggered && !earlySent { capture.append(contentsOf: channel) }
        quietFrames = rms < offThreshold ? quietFrames + channel.count : 0
        let elapsed = frames - onset
        if elapsed >= rate {
            let pcm = Array(ring[cursor...]) + Array(ring[..<cursor])
            onCandidate?(Candidate(id: activeID, phase: "fallback", pcm: pcm, startFrame: frames - ring.count, onsetFrame: onset, emitFrame: frames, sourceStartFrame: frames-ring.count, sourceEndFrame: frames, sourceSamples: pcm.count, destinationOffset: 0))
            clear(); cooldown = frames + rounded(0.4)
        } else if elapsed >= rounded(0.2) && quietFrames >= rounded(0.1) && !earlySent {
            var pcm = [Float](repeating: 0, count: ring.count)
            let offset = (pcm.count - capture.count) / 2
            pcm.replaceSubrange(offset..<(offset + capture.count), with: capture)
            let candidate = Candidate(id: activeID, phase: "early", pcm: pcm, startFrame: sourceStart-offset, onsetFrame: onset, emitFrame: frames, sourceStartFrame: sourceStart, sourceEndFrame: frames, sourceSamples: capture.count, destinationOffset: offset)
            earlySent = true; capture.removeAll(keepingCapacity: true); onCandidate?(candidate)
        }
    }
}

public enum FloatWAV {
    public static func encode(_ samples: [Float], rate: Int) -> Data {
        var data = Data()
        func ascii(_ string: String) { data.append(contentsOf: string.utf8) }
        func u16(_ value: UInt16) { var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
        func u32(_ value: UInt32) { var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
        ascii("RIFF"); u32(UInt32(50 + samples.count * 4)); ascii("WAVEfmt ")
        u32(18); u16(3); u16(1); u32(UInt32(rate)); u32(UInt32(rate * 4)); u16(4); u16(32); u16(0)
        ascii("fact"); u32(4); u32(UInt32(samples.count)); ascii("data"); u32(UInt32(samples.count * 4))
        for sample in samples { u32(sample.bitPattern) }
        return data
    }
}

public struct Prediction: Decodable {
    public struct Quality: Decodable { public let event_start_seconds: Double?, event_end_seconds: Double? }
    public let prediction: String?, reason: String?, quality: Quality?, processing_ms: Double?, error: String?
}

public struct EventGate {
    public private(set) var lastEnd = -Double.infinity
    private var accepted = Set<Int>()
    public init() {}
    public mutating func accept(_ result: Prediction, candidate: Candidate, rate: Int) -> Bool {
        guard result.reason == "accepted", ["up", "down"].contains(result.prediction ?? ""), !accepted.contains(candidate.id),
              let start = result.quality?.event_start_seconds, let end = result.quality?.event_end_seconds,
              start.isFinite, end.isFinite, end >= start else { return false }
        let absoluteStart = Double(candidate.startFrame) / Double(rate) + start
        let absoluteEnd = Double(candidate.startFrame) / Double(rate) + end
        guard absoluteStart > lastEnd + 0.1 else { return false }
        lastEnd = absoluteEnd; accepted.insert(candidate.id)
        if accepted.count > 32 { accepted = [candidate.id] }
        return true
    }
}
