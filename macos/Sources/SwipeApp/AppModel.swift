import AppKit
import AVFoundation
import Combine
import SwipeCore

@MainActor final class AppModel: ObservableObject {
    @Published var up = Mapping(action: .volumeUp, shortcut: .up) { didSet { save() } }
    @Published var down = Mapping(action: .volumeDown, shortcut: .down) { didSet { save() } }
    @Published var volumePercent = 6.0 { didSet { UserDefaults.standard.set(volumePercent, forKey: "volumePercent") } }
    @Published private(set) var listening = false
    @Published private(set) var starting = false
    @Published private(set) var status = "Paused"
    @Published private(set) var detail = "Start listening when you’re ready."
    @Published private(set) var result = "Ready for a swipe"
    @Published private(set) var resultIcon = "arrow.up.arrow.down"
    @Published private(set) var level: Double = 0
    @Published private(set) var error: String?
    @Published private(set) var micName = "Built-in microphone"
    @Published var accessibility = SystemActions.accessibilityAllowed
    @Published var project: URL?
    @Published var recordingDirection: String?
    var stateChanged: (() -> Void)?
    private let microphone = Microphone(), worker = Recognizer()
    private var generation = 0, busy = false, queued: (Candidate, Int)?, gate = EventGate(), acceptedID: Int?
    private var notifications: [NSObjectProtocol] = []
    var active: Bool { listening || starting }
    var needsAccessibility: Bool { up.action.needsAccessibility || down.action.needsAccessibility }
    var disruptive: Bool { up.action.pausesListening || down.action.pausesListening }
    init() {
        let defaults = UserDefaults.standard
        if let data = defaults.data(forKey: "up"), let value = try? JSONDecoder().decode(Mapping.self, from: data) { up = value }
        if let data = defaults.data(forKey: "down"), let value = try? JSONDecoder().decode(Mapping.self, from: data) { down = value }
        if defaults.object(forKey: "volumePercent") != nil { volumePercent = min(25, max(1, defaults.double(forKey: "volumePercent"))) }
        if let path = defaults.string(forKey: "project") { project = URL(fileURLWithPath: path) }
        else if let url = Bundle.main.url(forResource: "runtime", withExtension: "json"), let data = try? Data(contentsOf: url), let config = try? JSONDecoder().decode([String:String].self, from: data), let path = config["project"] { project = URL(fileURLWithPath: path) }
        notifications.append(NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: .main) { [weak self] _ in Task { @MainActor in self?.stop(detail: "Paused for sleep. Start listening again after waking.") } })
        notifications.append(NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in Task { @MainActor in self?.accessibility = SystemActions.accessibilityAllowed } })
        worker.onFailure = { [weak self] message in self?.fail(message) }
    }
    private func save() {
        UserDefaults.standard.set(try? JSONEncoder().encode(up), forKey: "up")
        UserDefaults.standard.set(try? JSONEncoder().encode(down), forKey: "down")
    }
    func chooseProject() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.message = "Choose the Swipe checkout containing swipe/, models/, and .venv/."; panel.prompt = "Use project"
        if panel.runModal() == .OK, let url = panel.url {
            stop(); project = url; UserDefaults.standard.set(url.path, forKey: "project")
        }
    }
    func allowAccessibility() { SystemActions.requestAccessibility(); accessibility = SystemActions.accessibilityAllowed }
    func start() {
        guard !active else { return }
        error = nil; recordingDirection = nil; accessibility = SystemActions.accessibilityAllowed
        guard !needsAccessibility || accessibility else { error = "Allow Accessibility access for Swipe before using this mapping."; return }
        guard let project else { error = "Choose your Swipe project folder to connect the local recognizer."; return }
        generation += 1; let epoch = generation
        starting = true; status = "Connecting"; detail = "Preparing the microphone and local recognizer…"; stateChanged?()
        Task { @MainActor in
            let granted: Bool
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: granted = true
            case .notDetermined: granted = await AVCaptureDevice.requestAccess(for: .audio)
            default: granted = false
            }
            guard generation == epoch, starting else { return }
            guard granted else { fail("Microphone access is off. Allow Swipe in System Settings → Privacy & Security → Microphone."); return }
            do {
                try worker.start(project: project) { [weak self] response in
                    guard let self, self.generation == epoch, self.starting else { return }
                    switch response {
                    case .failure(let error): self.fail(error.localizedDescription)
                    case .success: self.beginCapture(epoch: epoch)
                    }
                }
            } catch { fail(error.localizedDescription) }
        }
    }
    private func beginCapture(epoch: Int) {
        gate = EventGate(); acceptedID = nil; busy = false; queued = nil
        microphone.onCandidate = { [weak self] candidate, rate in
            DispatchQueue.main.async { guard let self, self.generation == epoch, self.listening else { return }; self.predict(candidate, rate: rate, epoch: epoch) }
        }
        microphone.onLevel = { [weak self] peak, warming in
            DispatchQueue.main.async {
                guard let self, self.generation == epoch, self.listening else { return }
                self.level = min(1, max(0, (20 * log10(Double(max(peak, 0.000001))) + 70) / 70))
                self.status = warming ? "Checking background" : "Listening"
                if !warming && self.detail == "Keep still for one second." { self.detail = "One swipe, then lift your finger and wait for the result." }
                self.stateChanged?()
            }
        }
        microphone.onFailure = { [weak self] message in DispatchQueue.main.async { guard let self, self.generation == epoch else { return }; self.fail(message) } }
        do {
            micName = try microphone.start(); starting = false; listening = true; status = "Checking background"
            detail = "Keep still for one second."; result = "Listening for a swipe"; resultIcon = "waveform"; stateChanged?()
        } catch { fail(error.localizedDescription) }
    }
    private func predict(_ candidate: Candidate, rate: Int, epoch: Int) {
        guard candidate.id != acceptedID else { return }
        if busy { if candidate.phase == "fallback" { queued = (candidate, rate) }; return }
        busy = true
        worker.predict(FloatWAV.encode(candidate.pcm, rate: rate)) { [weak self] response in
            guard let self, self.generation == epoch, self.listening else { return }
            self.busy = false
            switch response {
            case .failure(let error): self.fail(error.localizedDescription); return
            case .success(let prediction):
                if prediction.reason == "accepted", ["up", "down"].contains(prediction.prediction ?? "") {
                    if candidate.phase == "early" { self.microphone.accept(candidate.id); self.acceptedID = candidate.id; if self.queued?.0.id == candidate.id { self.queued = nil } }
                    if self.gate.accept(prediction, candidate: candidate, rate: rate) {
                        let isUp = prediction.prediction == "up", mapping = isUp ? self.up : self.down
                        // Destructive-to-session actions pause capture before posting, so wake/unlock never repeats them.
                        if mapping.action.pausesListening { self.stop(detail: "Paused after \(mapping.action.title.lowercased()). Start listening to resume.") }
                        do {
                            let action = try SystemActions.perform(mapping, volumeStep: Float(self.volumePercent / 100))
                            self.result = "\(isUp ? "Up" : "Down") · \(action)"; self.resultIcon = isUp ? "arrow.up" : "arrow.down"
                        } catch { self.fail(error.localizedDescription) }
                    }
                } else if candidate.phase == "fallback" {
                    self.result = prediction.prediction == "background" ? "Background sound ignored" : "Uncertain · no action"
                    self.resultIcon = "waveform"
                }
            }
            if let next = self.queued, self.generation == epoch, self.listening { self.queued = nil; self.predict(next.0, rate: next.1, epoch: epoch) }
        }
    }
    func stop(detail: String = "Microphone released. Your mappings are saved.") {
        generation += 1; listening = false; starting = false; busy = false; queued = nil; acceptedID = nil
        microphone.stop(); worker.stop(); level = 0; status = "Paused"; self.detail = detail; stateChanged?()
    }
    private func fail(_ message: String) { stop(); error = message; result = "Swipe paused"; resultIcon = "exclamationmark.circle" }
    func assignShortcut(_ shortcut: Shortcut, direction: String) {
        if direction == "up" { up.shortcut = shortcut } else { down.shortcut = shortcut }; recordingDirection = nil
    }
}
