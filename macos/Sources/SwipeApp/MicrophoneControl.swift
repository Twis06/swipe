import Foundation
import SwipeCore

protocol MicrophoneDriver: AnyObject, Sendable {
    var onCandidate: ((Candidate, Int) -> Void)? { get set }
    var onLevel: ((Float, Bool) -> Void)? { get set }
    var onFailure: ((String) -> Void)? { get set }
    func start() throws -> String
    func stop()
    func accept(_ id: Int)
}

/// The driver and all potentially blocking device operations belong to this serial queue.
/// Cancelling immediately suppresses callbacks; disposal never blocks a menu interaction.
@MainActor final class Microphone {
    private let control = DispatchQueue(label: "swipe.audio-control", qos: .userInitiated)
    private let driver: MicrophoneDriver
    private var token = OperationToken() // start/stop are called only on the main actor
    init(driver: MicrophoneDriver = AudioQueueMicrophoneDriver()) { self.driver = driver }

    @MainActor func start(candidate: @escaping (Candidate, Int) -> Void,
                         level: @escaping (Float, Bool) -> Void,
                         failure: @escaping (String) -> Void,
                         completion: @escaping (Result<String, Error>) -> Void) {
        token.cancel(); let current = OperationToken(); token = current
        control.async { [driver] in
            guard !current.isCancelled else { return }
            driver.onCandidate = { item, rate in if !current.isCancelled { candidate(item, rate) } }
            driver.onLevel = { value, warming in if !current.isCancelled { level(value, warming) } }
            driver.onFailure = { message in if !current.isCancelled { failure(message) } }
            let outcome = Result { try driver.start() }
            if current.isCancelled { driver.stop(); return }
            DispatchQueue.main.async { if !current.isCancelled { completion(outcome) } }
        }
    }
    @MainActor func stop(completion: @escaping () -> Void = {}) {
        token.cancel()
        control.async { [driver] in driver.stop(); DispatchQueue.main.async(execute: completion) }
    }
    @MainActor func accept(_ id: Int) {
        let current = token
        control.async { [driver] in if !current.isCancelled { driver.accept(id) } }
    }
}
