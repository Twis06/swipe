import SwiftUI
import AppKit
import AVFoundation
import SwipeCore

struct ContentView: View {
    @ObservedObject var model: AppModel
    @State private var advanced = false
    @Environment(\.colorScheme) private var colorScheme
    private var accent: Color { colorScheme == .dark ? Color(red: 0.40, green: 0.80, blue: 0.64) : Color(red: 0.08, green: 0.40, blue: 0.30) }
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Swipe").font(.system(size: 28, weight: .semibold))
                    Text("Your Mac. Another input.").font(.subheadline).foregroundStyle(.secondary)
                }
                Spacer()
                Label(model.status, systemImage: model.active ? "mic.fill" : "pause.circle")
                    .font(.callout.weight(.medium)).foregroundStyle(model.listening ? accent : .secondary)
                    .padding(.top, 8)
            }
            VStack(alignment: .leading, spacing: 16) {
                mappingRow(direction: "up", title: "Swipe up", detail: "Toward the screen", mapping: $model.up)
                Divider()
                mappingRow(direction: "down", title: "Swipe down", detail: "Toward the palm rest", mapping: $model.down)
            }.padding(18).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Volume step").font(.callout)
                    Spacer()
                    Text("\(model.volumePercent, specifier: "%.0f")%").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                }
                Slider(value: $model.volumePercent, in: 1...25, step: 1).tint(accent).disabled(model.active).accessibilityLabel("Volume change per swipe")
            }
            VStack(alignment: .leading, spacing: 9) {
                Label(model.result, systemImage: model.resultIcon).font(.callout.weight(.medium)).lineLimit(2)
                MicrophoneLevelView(meter: model.meter).tint(accent)
                Text(model.detail).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let error = model.error {
                Label(error, systemImage: "exclamationmark.circle").font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                if AVCaptureDevice.authorizationStatus(for: .audio) == .denied {
                    Button("Open microphone settings") { NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!) }.buttonStyle(.link)
                }
            }
            if model.needsAccessibility && !model.accessibility {
                HStack(alignment: .top) {
                    Text("Shortcuts and lock screen need Accessibility access for Swipe.").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Allow access") { model.allowAccessibility() }
                }
            }
            if model.disruptive {
                Text("Lock and sleep pause the microphone. Listening resumes when you return.").font(.callout).foregroundStyle(.secondary)
            }
            HStack {
                Button(model.wantsListening ? "Pause listening" : "Start listening") { model.wantsListening ? model.stop() : model.start() }
                    .buttonStyle(.borderedProminent).tint(accent).controlSize(.large)
                    .disabled(model.recordingDirection != nil)
                Spacer()
                Text("Audio stays on this Mac.").font(.caption).foregroundStyle(.secondary)
            }
            Divider()
            DisclosureGroup("Setup", isExpanded: $advanced) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Use the left speaker grille and the built-in microphone. One movement, then lift your finger.").font(.callout).foregroundStyle(.secondary)
                    HStack { Text(model.project == nil ? "Recognizer not configured" : "Local recognizer configured").font(.callout); Spacer(); Button("Choose project…") { model.chooseProject() }.disabled(model.active) }
                    Text("Experimental · Developed on a 14-inch M3 Pro MacBook Pro.").font(.caption).foregroundStyle(.secondary)
                }.padding(.top, 10)
            }.font(.callout)
        }
        .padding(26).frame(width: 490).background(Color(nsColor: .windowBackgroundColor))
        .onDisappear { model.recordingDirection = nil }
    }
    private func mappingRow(direction: String, title: String, detail: String, mapping: Binding<Mapping>) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: direction == "up" ? "arrow.up" : "arrow.down").font(.system(size: 22, weight: .medium)).foregroundStyle(accent).frame(width: 26)
                VStack(alignment: .leading, spacing: 3) { Text(title).font(.callout.weight(.semibold)); Text(detail).font(.caption).foregroundStyle(.secondary) }
                Spacer(minLength: 12)
                Picker(title, selection: mapping.action) {
                    ForEach(SwipeAction.allCases) { action in Label(action.title, systemImage: action.icon).tag(action) }
                }.labelsHidden().frame(width: 185).disabled(model.active).accessibilityLabel("\(title) action")
            }
            if mapping.wrappedValue.action == .shortcut {
                HStack {
                    Text("To the app in front").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if model.recordingDirection == direction {
                        ShortcutRecorder { shortcut in model.assignShortcut(shortcut, direction: direction) } cancel: { model.recordingDirection = nil }
                            .frame(width: 185, height: 28)
                    } else {
                        Button(mapping.wrappedValue.shortcut.label + "  ·  Record") { model.recordingDirection = direction }
                            .disabled(model.active).accessibilityLabel("Record \(title.lowercased()) shortcut")
                            .accessibilityValue(mapping.wrappedValue.shortcut.label)
                    }
                }
            }
        }
    }
}
struct ShortcutRecorder: NSViewRepresentable {
    let captured: (Shortcut) -> Void
    let cancel: () -> Void
    func makeNSView(context: Context) -> KeyCaptureView { let view = KeyCaptureView(); view.captured = captured; view.cancel = cancel; return view }
    func updateNSView(_ view: KeyCaptureView, context: Context) { view.captured = captured; view.cancel = cancel }
}
final class KeyCaptureView: NSTextField {
    var captured: ((Shortcut) -> Void)?, cancel: (() -> Void)?
    override var acceptsFirstResponder: Bool { true }
    init() {
        super.init(frame: .zero); stringValue = "Press shortcut… (Esc cancels)"; isEditable = false; isSelectable = false; alignment = .center; bezelStyle = .roundedBezel
        setAccessibilityLabel("Press a key or shortcut. Escape cancels.")
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); DispatchQueue.main.async { [weak self] in guard let self else { return }; self.window?.makeFirstResponder(self) } }
    override func performKeyEquivalent(with event: NSEvent) -> Bool { capture(event); return true }
    override func keyDown(with event: NSEvent) { capture(event) }
    private func capture(_ event: NSEvent) {
        if event.keyCode == 53 { cancel?(); return }
        let mask: NSEvent.ModifierFlags = [.command, .control, .option, .shift]
        let flags = event.modifierFlags.intersection(mask)
        let specials: [UInt16:String] = [123:"←",124:"→",125:"↓",126:"↑",49:"Space",36:"Return",48:"Tab",51:"Delete",117:"Forward delete",115:"Home",119:"End",116:"Page up",121:"Page down"]
        let glyph = specials[event.keyCode] ?? event.charactersIgnoringModifiers?.uppercased() ?? "Key \(event.keyCode)"
        let prefix = [(NSEvent.ModifierFlags.control,"⌃"),(.option,"⌥"),(.shift,"⇧"),(.command,"⌘")].filter { flags.contains($0.0) }.map { $0.1 }.joined()
        captured?(Shortcut(keyCode: event.keyCode, modifiers: UInt64(flags.rawValue), label: prefix + glyph))
    }
}

private struct MicrophoneLevelView: View {
    @ObservedObject var meter: AudioMeter
    var body: some View { ProgressView(value: meter.value).accessibilityLabel("Microphone level") }
}
