import Foundation
public enum SwipeAction: String, CaseIterable, Codable, Identifiable, Sendable {
    case volumeUp, volumeDown, mute, lockScreen, sleep, displaySleep, shortcut, none
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .volumeUp: return "Volume up"
        case .volumeDown: return "Volume down"
        case .mute: return "Mute / unmute"
        case .lockScreen: return "Lock screen"
        case .sleep: return "Sleep Mac"
        case .displaySleep: return "Turn display off"
        case .shortcut: return "Keyboard shortcut"
        case .none: return "Do nothing"
        }
    }
    public var icon: String {
        switch self {
        case .volumeUp: return "speaker.wave.3"
        case .volumeDown: return "speaker.wave.1"
        case .mute: return "speaker.slash"
        case .lockScreen: return "lock"
        case .sleep: return "moon"
        case .displaySleep: return "display"
        case .shortcut: return "command"
        case .none: return "minus"
        }
    }
    public var needsAccessibility: Bool { self == .lockScreen || self == .shortcut }
    public var pausesListening: Bool { [.lockScreen, .sleep, .displaySleep].contains(self) }
}
public struct Shortcut: Codable, Equatable, Sendable {
    public var keyCode: UInt16
    public var modifiers: UInt64
    public var label: String
    public init(keyCode: UInt16, modifiers: UInt64 = 0, label: String) { self.keyCode = keyCode; self.modifiers = modifiers; self.label = label }
    public static let up = Shortcut(keyCode: 126, label: "↑")
    public static let down = Shortcut(keyCode: 125, label: "↓")
}
public struct Mapping: Codable, Equatable, Sendable {
    public var action: SwipeAction
    public var shortcut: Shortcut
    public init(action: SwipeAction, shortcut: Shortcut) { self.action = action; self.shortcut = shortcut }
}
