import Foundation
import Darwin
import SwipeCore

struct SwipeError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

/// One private child process, one request at a time; never opens a network port.
@MainActor final class Recognizer {
    private var process: Process?
    private var input: FileHandle?
    private let queue = DispatchQueue(label: "swipe.recognizer", qos: .userInitiated)
    private var timeout: DispatchWorkItem?
    private var operation = OperationToken()
    var onFailure: ((String) -> Void)?
    func start(project: URL, ready: @escaping (Result<Void, Error>) -> Void) throws {
        stop()
        let python = project.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: project.appendingPathComponent("swipe/native_worker.py").path) else {
            throw SwipeError("Choose the Swipe project folder with its .venv environment. See macos/README.md for setup.")
        }
        let child = Process(), stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        child.executableURL = python; child.currentDirectoryURL = project
        child.arguments = ["-u", "-m", "swipe.native_worker"]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = project.path; env["PYTHONNOUSERSITE"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"; env["OMP_NUM_THREADS"] = "1"
        child.environment = env; child.standardInput = stdin; child.standardOutput = stdout; child.standardError = stderr
        // Drain diagnostic output without storing microphone data or exposing user paths in UI.
        stderr.fileHandleForReading.readabilityHandler = { handle in
            if handle.availableData.isEmpty { handle.readabilityHandler = nil }
        }
        child.terminationHandler = { [weak self] child in
            DispatchQueue.main.async {
                guard let self, self.process === child else { return }
                self.onFailure?("The recognizer stopped. Start listening to reconnect.")
            }
        }
        let current = OperationToken(); operation = current
        process = child; input = stdin.fileHandleForWriting
        scheduleTimeout(child, seconds: 15)
        let output = stdout.fileHandleForReading
        queue.async { [weak self] in
            let reader = JSONLineReader(output)
            let result: Result<Void, Error>
            do {
                guard !current.isCancelled else { return }
                try child.run()
                guard !current.isCancelled else { child.terminate(); return }
                let info = try JSONSerialization.jsonObject(with: reader.read()) as? [String: Any]
                guard info?["ready"] as? Bool == true else { throw SwipeError(info?["error"] as? String ?? "The local model could not load.") }
                result = .success(())
            } catch { result = .failure(error) }
            DispatchQueue.main.async {
                guard let self, self.process === child else { return }
                self.timeout?.cancel(); self.reader = reader; ready(result)
            }
        }
    }
    private var reader: JSONLineReader?
    func predict(_ data: Data, completion: @escaping (Result<Prediction, Error>) -> Void) {
        send(encode: { data }, completion: completion)
    }
    func predict(samples: [Float], rate: Int, completion: @escaping (Result<Prediction, Error>) -> Void) {
        send(encode: { FloatWAV.encode(samples, rate: rate) }, completion: completion)
    }
    private func send(encode: @escaping () -> Data, completion: @escaping (Result<Prediction, Error>) -> Void) {
        guard let child = process, let input, let reader else { completion(.failure(SwipeError("The recognizer is not connected."))); return }
        let current = operation
        scheduleTimeout(child, seconds: 6)
        queue.async { [weak self] in
            let result: Result<Prediction, Error>
            do {
                guard !current.isCancelled else { return }
                let data = encode()
                var size = UInt32(data.count).littleEndian
                var framed = withUnsafeBytes(of: &size) { Data($0) }; framed.append(data)
                try input.write(contentsOf: framed)
                let prediction = try JSONDecoder().decode(Prediction.self, from: reader.read())
                if let error = prediction.error { throw SwipeError(error) }
                result = .success(prediction)
            } catch { result = .failure(error) }
            DispatchQueue.main.async {
                guard let self, self.process === child else { return }
                self.timeout?.cancel(); completion(result)
            }
        }
    }
    private func scheduleTimeout(_ child: Process, seconds: Double) {
        timeout?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.process === child else { return }
            self.stop(); self.onFailure?("The local recognizer timed out. Start listening to reconnect.")
        }
        timeout = work; DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: work)
    }
    func stop() {
        operation.cancel()
        timeout?.cancel(); timeout = nil
        let old = process; process = nil; reader = nil
        try? input?.close(); input = nil
        if old?.isRunning == true { old?.terminate() }
    }
}
// All reads and buffer mutations are confined to Recognizer.queue.
private final class JSONLineReader: @unchecked Sendable {
    let handle: FileHandle
    var buffer = Data()
    init(_ handle: FileHandle) { self.handle = handle }
    func read() throws -> Data {
        while true {
            if let newline = buffer.firstIndex(of: 10) {
                let line = Data(buffer[..<newline]); buffer.removeSubrange(...newline); return line
            }
            guard buffer.count < 1_000_000 else { throw SwipeError("The recognizer returned an oversized response.") }
            // FileHandle.read(upToCount:) can wait to fill its buffer on a pipe.
            // A single POSIX read returns the available bytes, including short JSON lines.
            var chunk = [UInt8](repeating: 0, count: 65536)
            let count = chunk.withUnsafeMutableBytes { Darwin.read(handle.fileDescriptor, $0.baseAddress, $0.count) }
            if count < 0 && errno == EINTR { continue }
            guard count > 0 else { throw SwipeError("The recognizer disconnected. Check the local Python environment.") }
            buffer.append(contentsOf: chunk.prefix(count))
        }
    }
}
