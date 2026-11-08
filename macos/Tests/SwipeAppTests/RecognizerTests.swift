import Testing
import Foundation
@testable import SwipeApp
import SwipeCore

final class RecognizerTests {
    @Test @MainActor func testRealWorkerRoundTripAndStop() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath: root.appendingPathComponent(".venv/bin/python").path) else {
            Issue.record("Integration test needs the local recognizer .venv. See macos/README.md."); return
        }
        let worker = Recognizer()
        defer { worker.stop() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            worker.onFailure = { message in continuation.resume(throwing: SwipeError(message)) }
            do { try worker.start(project: root) { continuation.resume(with: $0) } }
            catch { continuation.resume(throwing: error) }
        }
        let prediction: Prediction = try await withCheckedThrowingContinuation { continuation in
            worker.onFailure = { message in continuation.resume(throwing: SwipeError(message)) }
            worker.predict(FloatWAV.encode([Float](repeating: 0, count: 96000), rate: 48000)) { continuation.resume(with: $0) }
        }
        worker.onFailure = nil
        #expect((prediction.reason) != ("accepted"))
        #expect((prediction.prediction) != ("up"))
        #expect((prediction.prediction) != ("down"))
        worker.stop()
        let afterStop: Result<Prediction, Error> = await withCheckedContinuation { continuation in
            worker.predict(Data()) { continuation.resume(returning: $0) }
        }
        if case .success = afterStop { Issue.record("Stopped worker must reject requests") }
    }
}
