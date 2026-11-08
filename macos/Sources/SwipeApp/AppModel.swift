import AppKit
import AVFoundation
import Combine
import SwipeCore

@MainActor final class AppModel: ObservableObject {
    @Published var up = Mapping(action: .volumeUp, shortcut: .up) { didSet { save() } }
    @Published var down = Mapping(action: .volumeDown, shortcut: .down) { didSet { save() } }
    @Published var volumePercent = 6.0 { didSet { preferences.set(volumePercent, forKey: "volumePercent") } }
    @Published private(set) var listening = false
    @Published private(set) var starting = false
    @Published private(set) var status = "Paused"
    @Published private(set) var detail = "Start listening when you’re ready."
    @Published private(set) var result = "Ready for a swipe"
    @Published private(set) var resultIcon = "arrow.up.arrow.down"
    let meter = AudioMeter()
    @Published private(set) var wantsListening = false
    @Published private(set) var error: String?
    @Published private(set) var micName = "Built-in microphone"
    @Published var accessibility = SystemActions.accessibilityAllowed
    @Published var project: URL?
    @Published var recordingDirection: String?
    var stateChanged: (() -> Void)?
    private let microphone: Microphone
    private let worker: Recognizer
    private let preferences: UserDefaults
    private var restoringPreferences = true
    private let audioPermission: () async -> Bool
    private var generation = 0, busy = false, queued: (Candidate, Int)?, gate = EventGate(), acceptedID: Int?
    private var notifications: [NSObjectProtocol] = []
    private var session = ListeningSession()
    private var sessionMonitor: SessionMonitor?
    private var resumeTask: Task<Void, Never>?
    private let actionQueue = DispatchQueue(label: "swipe.actions", qos: .userInitiated)
    private var actionGeneration = OperationToken()
    var active: Bool { listening || starting }
    var needsAccessibility: Bool { up.action.needsAccessibility || down.action.needsAccessibility }
    var disruptive: Bool { up.action.pausesListening || down.action.pausesListening }
    init(defaults: UserDefaults = .standard, microphone: Microphone? = nil,
         workspace: NotificationCenter = NSWorkspace.shared.notificationCenter,
         distributed: NotificationCenter = DistributedNotificationCenter.default(),
         audioPermission: (() async -> Bool)? = nil) {
        preferences = defaults; self.microphone = microphone ?? Microphone(); worker = Recognizer()
        self.audioPermission = audioPermission ?? Self.requestMicrophonePermission
        if let data = defaults.data(forKey: "up"), let value = try? JSONDecoder().decode(Mapping.self, from: data) { up = value }
        if let data = defaults.data(forKey: "down"), let value = try? JSONDecoder().decode(Mapping.self, from: data) { down = value }
        if defaults.object(forKey: "volumePercent") != nil { volumePercent = min(25, max(1, defaults.double(forKey: "volumePercent"))) }
        if let path = defaults.string(forKey: "project") { project = URL(fileURLWithPath: path) }
        else if let url = Bundle.main.url(forResource: "runtime", withExtension: "json"), let data = try? Data(contentsOf: url), let config = try? JSONDecoder().decode([String:String].self, from: data), let path = config["project"] { project = URL(fileURLWithPath: path) }
        sessionMonitor = SessionMonitor(workspace: workspace, distributed: distributed) { [weak self] blocker, present in
            Task { @MainActor in self?.sessionChanged(blocker, present: present) }
        }
        notifications.append(NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { [weak self] _ in Task { @MainActor in self?.accessibility = SystemActions.accessibilityAllowed } })
        worker.onFailure = { [weak self] message in self?.fail(message) }
        restoringPreferences = false
    }
    private static func requestMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }
    private func save() {
        guard !restoringPreferences else { return }
        preferences.set(try? JSONEncoder().encode(up), forKey: "up")
        preferences.set(try? JSONEncoder().encode(down), forKey: "down")
    }
    func chooseProject() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.message = "Choose the Swipe checkout containing swipe/, models/, and .venv/."; panel.prompt = "Use project"
        if panel.runModal() == .OK, let url = panel.url {
            stop(); project = url; preferences.set(url.path, forKey: "project")
        }
    }
    func allowAccessibility() { SystemActions.requestAccessibility(); accessibility = SystemActions.accessibilityAllowed }
    func start() {
        session.request(); wantsListening = true
        if session.shouldListen { startCapture() } else { showSuspension() }
    }
    private func startCapture() {
        guard session.shouldListen, !active else { return }
        error = nil; recordingDirection = nil; accessibility = SystemActions.accessibilityAllowed
        guard !needsAccessibility || accessibility else { fail("Allow Accessibility access for Swipe before using this mapping."); return }
        guard let project else { fail("Choose your Swipe project folder to connect the local recognizer."); return }
        generation += 1; let epoch = generation
        starting = true; status = "Connecting"; detail = "Preparing the microphone and local recognizer…"; stateChanged?()
        Task { @MainActor in
            let granted = await audioPermission()
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
        actionGeneration = OperationToken()
        microphone.start(candidate: { [weak self] candidate, rate in
            DispatchQueue.main.async { guard let self, self.generation == epoch, self.listening else { return }; self.predict(candidate, rate: rate, epoch: epoch) }
        }, level: { [weak self] peak, warming in
            DispatchQueue.main.async {
                guard let self, self.generation == epoch, self.listening else { return }
                self.meter.value = min(1, max(0, (20 * log10(Double(max(peak, 0.000001))) + 70) / 70))
                let next = warming ? "Checking background" : "Listening"
                if self.status != next { self.status = next; self.stateChanged?() }
                if !warming && self.detail == "Keep still for one second." { self.detail = "One swipe, then lift your finger and wait for the result." }
            }
        }, failure: { [weak self] message in
            DispatchQueue.main.async { guard let self, self.generation == epoch else { return }; self.fail(message) }
        }, completion: { [weak self] response in
            guard let self, self.generation == epoch, self.starting else { return }
            switch response {
            case .success(let name):
                self.micName = name; self.starting = false; self.listening = true; self.status = "Checking background"
                self.detail = "Keep still for one second."; self.result = "Listening for a swipe"; self.resultIcon = "waveform"; self.stateChanged?()
            case .failure(let error): self.fail(error.localizedDescription)
            }
        })
    }
    private func predict(_ candidate: Candidate, rate: Int, epoch: Int) {
        guard candidate.id != acceptedID else { return }
        if busy { if candidate.phase == "fallback" { queued = (candidate, rate) }; return }
        busy = true
        worker.predict(samples: candidate.pcm, rate: rate) { [weak self] response in
            guard let self, self.generation == epoch, self.listening else { return }
            self.busy = false
            switch response {
            case .failure(let error): self.fail(error.localizedDescription); return
            case .success(let prediction):
                if prediction.reason == "accepted", ["up", "down"].contains(prediction.prediction ?? "") {
                    if candidate.phase == "early" { self.microphone.accept(candidate.id); self.acceptedID = candidate.id; if self.queued?.0.id == candidate.id { self.queued = nil } }
                    if self.gate.accept(prediction, candidate: candidate, rate: rate) {
                        let isUp = prediction.prediction == "up", mapping = isUp ? self.up : self.down
                        self.perform(mapping, isUp: isUp)
                    }
                } else if candidate.phase == "fallback" {
                    self.result = prediction.prediction == "background" ? "Background sound ignored" : "Uncertain · no action"
                    self.resultIcon = "waveform"
                }
            }
            if let next = self.queued, self.generation == epoch, self.listening { self.queued = nil; self.predict(next.0, rate: next.1, epoch: epoch) }
        }
    }
    private func perform(_ mapping: Mapping, isUp: Bool) {
        if mapping.action.pausesListening {
            let blocker: ListeningSession.Blocker = mapping.action == .lockScreen ? .locked : (mapping.action == .sleep ? .sleeping : .displayOff)
            session.prepare(blocker)
            let suspendedEpoch = generation + 1
            haltCapture(detail: "Listening will resume when you return.") { [weak self] in
                guard let self, self.generation == suspendedEpoch, self.wantsListening else { return }
                self.actionGeneration = OperationToken()
                self.runAction(mapping, isUp: isUp)
            }
            showSuspension()
        } else { runAction(mapping, isUp: isUp) }
    }
    private func runAction(_ mapping: Mapping, isUp: Bool) {
        let token = actionGeneration, epoch = generation, step = Float(volumePercent / 100)
        actionQueue.async { [weak self] in
            guard !token.isCancelled else { return }
            let response = Result { try SystemActions.perform(mapping, volumeStep: step) }
            DispatchQueue.main.async {
                guard let self, !token.isCancelled, self.generation == epoch else { return }
                switch response {
                case .success(let action):
                    self.result = "\(isUp ? "Up" : "Down") · \(action)"; self.resultIcon = isUp ? "arrow.up" : "arrow.down"
                case .failure(let error): self.fail(error.localizedDescription)
                }
            }
        }
    }
    private func sessionChanged(_ blocker: ListeningSession.Blocker, present: Bool) {
        session.set(blocker, present: present)
        resumeTask?.cancel(); resumeTask = nil
        if !session.shouldListen {
            if active { haltCapture(detail: "Listening will resume when you return.") }
            if wantsListening { showSuspension() }
        } else if !active {
            status = "Resuming"; detail = "Reconnecting the built-in microphone…"; stateChanged?()
            resumeTask = Task { @MainActor [weak self] in
                do { try await Task.sleep(for: .milliseconds(300)) } catch { return }
                guard let self, self.session.shouldListen, !self.active else { return }
                self.startCapture()
            }
        }
    }
    private func showSuspension() {
        if session.waitingFor.contains(.locked) { status = "Waiting for unlock" }
        else if session.waitingFor.contains(.sleeping) { status = "Sleeping" }
        else if session.waitingFor.contains(.displayOff) { status = "Display off" }
        else { status = "Session inactive" }
        detail = "Listening will resume when you return. Pause to cancel."; stateChanged?()
    }
    func stop(detail: String = "Microphone released. Your mappings are saved.", completion: @escaping () -> Void = {}) {
        session.pause(); wantsListening = false
        resumeTask?.cancel(); resumeTask = nil
        haltCapture(detail: detail, completion: completion)
    }
    private func haltCapture(detail: String, completion: @escaping () -> Void = {}) {
        generation += 1; actionGeneration.cancel()
        listening = false; starting = false; busy = false; queued = nil; acceptedID = nil
        meter.value = 0; status = "Paused"; self.detail = detail; stateChanged?()
        worker.stop(); microphone.stop(completion: completion)
    }
    private func fail(_ message: String) { stop(); error = message; result = "Swipe paused"; resultIcon = "exclamationmark.circle" }
    func assignShortcut(_ shortcut: Shortcut, direction: String) {
        if direction == "up" { up.shortcut = shortcut } else { down.shortcut = shortcut }; recordingDirection = nil
    }
}
