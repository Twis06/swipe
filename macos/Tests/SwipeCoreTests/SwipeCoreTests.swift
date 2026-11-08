import Testing
import Foundation
import CryptoKit
@testable import SwipeCore

final class SwipeCoreTests {
    @Test func testCaptureMatchesShippedWorkletAtBothRates() throws {
        let url = Bundle.module.url(forResource: "capture-fixtures", withExtension: "json")!
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [[String: Any]]
        for fixture in cases {
            let rate = fixture["rate"] as! Int, accept = fixture["acceptEarly"] as! Bool
            let capture = GestureCapture(rate: rate), expected = fixture["candidates"] as! [[String: Any]]
            var actual: [Candidate] = []
            capture.onCandidate = { candidate in actual.append(candidate) }
            let warm = Int((Double(rate) * 1.2).rounded()), tone = Int((Double(rate) * 0.1).rounded())
            var offset = 0
            while offset < rate * 4 {
                let limit = min(offset + 128, rate * 4)
                let block: [Float] = (offset..<limit).map { i in
                    guard i >= warm && i < warm + tone else { return 0 }
                    let value = ((i - warm) % 29) - 14
                    return Float(value) / 16
                }
                capture.process(block)
                if accept, let early = actual.last, early.phase == "early" { capture.accept(early.id) }
                offset += block.count
            }
            #expect((actual.count) == (expected.count))
            for (candidate, truth) in zip(actual, expected) {
                #expect((candidate.phase) == (truth["phase"] as? String))
                #expect((candidate.startFrame) == (truth["startFrame"] as? Int))
                #expect((candidate.onsetFrame) == (truth["onsetFrame"] as? Int))
                #expect((candidate.emitFrame) == (truth["emitFrame"] as? Int))
                #expect((candidate.destinationOffset) == (truth["destinationOffset"] as? Int))
                #expect((candidate.sourceSamples) == (truth["sourceSamples"] as? Int))
                let bytes = candidate.pcm.withUnsafeBytes { Data($0) }
                #expect((SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()) == (truth["sha256"] as? String))
            }
        }
    }
    @Test func testStopAndStaleAcknowledgementCannotCancelNewEvent() {
        let capture = GestureCapture(rate: 48000); var values: [Candidate] = []
        capture.onCandidate = { values.append($0) }
        for _ in 0..<450 { capture.process([Float](repeating: 0, count: 128)) }
        for _ in 0..<40 { capture.process([Float](repeating: 0.1, count: 128)) }
        for _ in 0..<100 { capture.process([Float](repeating: 0, count: 128)) }
        #expect((values.first?.phase) == ("early"))
        capture.accept(999)
        for _ in 0..<400 { capture.process([Float](repeating: 0, count: 128)) }
        #expect((values.last?.phase) == ("fallback"))
        capture.stop(); let count = values.count
        for _ in 0..<1000 { capture.process([Float](repeating: 0.1, count: 128)) }
        #expect((values.count) == (count))
    }
    @Test func testFloatWAVPreservesHeadroomAndSignedZero() {
        let samples: [Float] = [0,-0.0,1.2,-2.5]
        let wav = FloatWAV.encode(samples, rate: 48000)
        #expect((wav.count) == (58 + samples.count*4))
        #expect((wav.subdata(in: 58..<wav.count)) == (samples.withUnsafeBytes { Data($0) }))
        #expect((String(data: wav.prefix(4), encoding: .utf8)) == ("RIFF"))
    }
    @Test func testGateNeverActsOnUncertainOrDuplicateResults() throws {
        let candidate = Candidate(id: 1, phase: "early", pcm: [], startFrame: 0, onsetFrame: 0, emitFrame: 0, sourceStartFrame: 0, sourceEndFrame: 0, sourceSamples: 0, destinationOffset: 0)
        func prediction(_ direction: String, _ reason: String) throws -> Prediction {
            try JSONDecoder().decode(Prediction.self, from: Data("{\"prediction\":\"\(direction)\",\"reason\":\"\(reason)\",\"quality\":{\"event_start_seconds\":0.8,\"event_end_seconds\":1.2}}".utf8))
        }
        var gate = EventGate()
        #expect(!(gate.accept(try prediction("uncertain", "ambiguous"), candidate: candidate, rate: 48000)))
        #expect(!(gate.accept(try prediction("up", "unfamiliar_sound"), candidate: candidate, rate: 48000)))
        #expect(gate.accept(try prediction("up", "accepted"), candidate: candidate, rate: 48000))
        #expect(!(gate.accept(try prediction("down", "accepted"), candidate: candidate, rate: 48000)))
        var fresh = EventGate(); #expect(fresh.accept(try prediction("up", "accepted"), candidate: candidate, rate: 48000))
    }
    @Test func testActionPermissionAndPausePolicy() {
        #expect(!(SwipeAction.volumeUp.needsAccessibility))
        #expect(SwipeAction.lockScreen.needsAccessibility)
        #expect(SwipeAction.shortcut.needsAccessibility)
        for action in [SwipeAction.lockScreen, .sleep, .displaySleep] { #expect(action.pausesListening) }
        #expect(!(SwipeAction.volumeDown.pausesListening))
        #expect((Set(SwipeAction.allCases.map(\.rawValue)).count) == (SwipeAction.allCases.count))
    }
}
