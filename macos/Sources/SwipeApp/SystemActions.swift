import AppKit
import CoreAudio
import IOKit.pwr_mgt
import SwipeCore

struct SystemActions {
    static var accessibilityAllowed: Bool { CGPreflightPostEventAccess() }
    static func requestAccessibility() { _ = CGRequestPostEventAccess() }
    static func perform(_ mapping: Mapping, volumeStep: Float) throws -> String {
        switch mapping.action {
        case .none: return "No action assigned"
        case .volumeUp: return try adjustVolume(volumeStep)
        case .volumeDown: return try adjustVolume(-volumeStep)
        case .mute: return try toggleMute()
        case .lockScreen:
            try post(Shortcut(keyCode: 12, modifiers: CGEventFlags.maskCommand.rawValue | CGEventFlags.maskControl.rawValue, label: "Lock screen"))
        case .shortcut: try post(mapping.shortcut)
        case .sleep:
            let connection = IOPMFindPowerManagement(mach_port_t(MACH_PORT_NULL))
            guard connection != 0 else { throw SwipeError("macOS power management is unavailable.") }
            defer { IOServiceClose(connection) }
            guard IOPMSleepSystem(connection) == kIOReturnSuccess else { throw SwipeError("macOS could not put this Mac to sleep.") }
        case .displaySleep:
            let task = Process(); task.executableURL = URL(fileURLWithPath: "/usr/bin/pmset"); task.arguments = ["displaysleepnow"]
            task.standardOutput = FileHandle.nullDevice; task.standardError = FileHandle.nullDevice
            try task.run(); task.waitUntilExit()
            guard task.terminationStatus == 0 else { throw SwipeError("macOS could not turn the display off.") }
        }
        return mapping.action == .shortcut ? "Sent \(mapping.shortcut.label)" : mapping.action.title
    }
    static func post(_ shortcut: Shortcut) throws {
        guard accessibilityAllowed else { throw SwipeError("Allow Accessibility access for Swipe to send shortcuts or lock the screen.") }
        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: shortcut.keyCode, keyDown: true), let up = CGEvent(keyboardEventSource: nil, virtualKey: shortcut.keyCode, keyDown: false) else { throw SwipeError("macOS could not create a keyboard event.") }
        down.flags = CGEventFlags(rawValue: shortcut.modifiers); up.flags = down.flags
        down.post(tap: .cghidEventTap); up.post(tap: .cghidEventTap)
    }
    static func defaultOutput() throws -> AudioDeviceID {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var device = AudioDeviceID(0), size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device) == noErr, device != 0 else { throw SwipeError("No output audio device is available.") }
        return device
    }
    static func channels(_ device: AudioDeviceID, selector: AudioObjectPropertySelector) -> [AudioObjectPropertyElement] {
        func writable(_ element: AudioObjectPropertyElement) -> Bool {
            var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioDevicePropertyScopeOutput, mElement: element)
            var writable = DarwinBoolean(false)
            return AudioObjectHasProperty(device, &address) && AudioObjectIsPropertySettable(device, &address, &writable) == noErr && writable.boolValue
        }
        if writable(kAudioObjectPropertyElementMain) { return [kAudioObjectPropertyElementMain] }
        return [1, 2].filter { writable($0) }
    }
    static func adjustVolume(_ delta: Float) throws -> String {
        let device = try defaultOutput(), elements = channels(device, selector: kAudioDevicePropertyVolumeScalar)
        guard !elements.isEmpty else { throw SwipeError("This output device does not offer software volume control. Choose MacBook speakers or another supported output.") }
        var changes: [(AudioObjectPropertyAddress, Float)] = []
        for element in elements {
            var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyVolumeScalar, mScope: kAudioDevicePropertyScopeOutput, mElement: element)
            var value: Float = 0, bytes = UInt32(MemoryLayout<Float>.size)
            guard AudioObjectGetPropertyData(device, &address, 0, nil, &bytes, &value) == noErr else { throw SwipeError("Could not read output volume.") }
            changes.append((address, value))
        }
        var mean: Float = 0
        for (var address, old) in changes {
            var value = min(1, max(0, old + delta))
            if AudioObjectSetPropertyData(device, &address, 0, nil, UInt32(MemoryLayout<Float>.size), &value) != noErr {
                for (var restoreAddress, var original) in changes { _ = AudioObjectSetPropertyData(device, &restoreAddress, 0, nil, UInt32(MemoryLayout<Float>.size), &original) }
                throw SwipeError("Could not change output volume.")
            }
            mean += value
        }
        if delta > 0 { try? setMute(device, muted: false) }
        return "Volume \(Int((mean / Float(elements.count) * 100).rounded()))%"
    }
    static func setMute(_ device: AudioDeviceID, muted: Bool) throws {
        let elements = channels(device, selector: kAudioDevicePropertyMute)
        guard !elements.isEmpty else { throw SwipeError("This audio device does not support mute control.") }
        for element in elements {
            var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyMute, mScope: kAudioDevicePropertyScopeOutput, mElement: element), value: UInt32 = muted ? 1 : 0
            guard AudioObjectSetPropertyData(device, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value) == noErr else { throw SwipeError("Could not change mute. Try another output device.") }
        }
    }
    static func toggleMute() throws -> String {
        let device = try defaultOutput(), elements = channels(device, selector: kAudioDevicePropertyMute)
        guard let first = elements.first else { throw SwipeError("This audio device does not support mute control.") }
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyMute, mScope: kAudioDevicePropertyScopeOutput, mElement: first)
        var value: UInt32 = 0, bytes = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &bytes, &value) == noErr else { throw SwipeError("Could not read mute state.") }
        try setMute(device, muted: value == 0); return value == 0 ? "Muted" : "Unmuted"
    }
}
