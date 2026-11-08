import AVFoundation
import CoreAudio
import AudioToolbox
import SwipeCore

final class Microphone {
    private var engine: AVAudioEngine?
    private let queue = DispatchQueue(label: "swipe.capture", qos: .userInitiated)
    private var capture: GestureCapture?
    private var pending: [Float] = [], frames = 0, peak: Float = 0, levelFrame = 0
    private var epoch = 0
    var onCandidate: ((Candidate, Int) -> Void)?
    var onLevel: ((Float, Bool) -> Void)?
    var onFailure: ((String) -> Void)?
    private var configObserver: NSObjectProtocol?

    func start() throws -> String {
        let device = try Self.builtInInput()
        let newEngine = AVAudioEngine(), input = newEngine.inputNode
        guard let unit = input.audioUnit else { throw SwipeError("The microphone is unavailable.") }
        var deviceID = device.id
        let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &deviceID, UInt32(MemoryLayout<AudioDeviceID>.size))
        guard status == noErr else { throw SwipeError("Could not select the built-in microphone. Reconnect and try again.") }
        let format = input.outputFormat(forBus: 0), rate = Int(format.sampleRate.rounded())
        guard [44100, 48000].contains(rate), format.channelCount > 0 else { throw SwipeError("Use a built-in microphone at 44.1 or 48 kHz in Audio MIDI Setup.") }
        epoch += 1; let current = epoch
        queue.sync {
            capture = GestureCapture(rate: rate); pending = []; frames = 0; peak = 0; levelFrame = 0
            capture?.onCandidate = { [weak self] candidate in self?.onCandidate?(candidate, rate) }
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self, let channels = buffer.floatChannelData else { return }
            let count = Int(buffer.frameLength), channelCount = Int(buffer.format.channelCount)
            var mono = [Float](repeating: 0, count: count)
            for channel in 0..<channelCount { for i in 0..<count { mono[i] += channels[channel][i] / Float(channelCount) } }
            let pcm = mono
            self.queue.async { [weak self] in
                guard let self, self.epoch == current else { return }
                guard pcm.allSatisfy({ $0.isFinite }) else { self.onFailure?("The microphone returned invalid audio. Start listening again."); return }
                self.pending.append(contentsOf: pcm)
                var offset = 0
                while self.pending.count - offset >= 128 {
                    let block = Array(self.pending[offset..<(offset + 128)])
                    self.capture?.process(block); self.frames += 128
                    self.peak = max(self.peak, block.map { abs($0) }.max() ?? 0); offset += 128
                    if self.frames - self.levelFrame >= rate / 10 {
                        self.onLevel?(self.peak, self.frames < rate); self.peak = 0; self.levelFrame = self.frames
                    }
                }
                if offset > 0 { self.pending.removeFirst(offset) }
            }
        }
        do { try newEngine.start() } catch { input.removeTap(onBus: 0); throw error }
        engine = newEngine
        configObserver = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: newEngine, queue: .main) { [weak self] _ in self?.onFailure?("The audio device changed. Start listening again to reconnect the built-in microphone.") }
        return device.name
    }
    func accept(_ id: Int) { queue.async { [weak self] in self?.capture?.accept(id) } }
    func stop() {
        if let configObserver { NotificationCenter.default.removeObserver(configObserver) }; configObserver = nil
        engine?.stop(); engine?.inputNode.removeTap(onBus: 0); engine = nil
        queue.sync { epoch += 1; capture?.stop(); capture = nil; pending = []; peak = 0 }
    }
    static func builtInInput() throws -> (id: AudioDeviceID, name: String) {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { throw SwipeError("Could not read audio devices.") }
        var devices = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &devices) == noErr else { throw SwipeError("Could not read audio devices.") }
        for device in devices {
            var transport: UInt32 = 0, bytes = UInt32(MemoryLayout<UInt32>.size)
            address.mSelector = kAudioDevicePropertyTransportType
            guard AudioObjectGetPropertyData(device, &address, 0, nil, &bytes, &transport) == noErr, transport == kAudioDeviceTransportTypeBuiltIn else { continue }
            address.mSelector = kAudioDevicePropertyStreams; address.mScope = kAudioDevicePropertyScopeInput
            var streamBytes: UInt32 = 0
            let hasInput = AudioObjectGetPropertyDataSize(device, &address, 0, nil, &streamBytes) == noErr && streamBytes > 0
            address.mScope = kAudioObjectPropertyScopeGlobal
            guard hasInput else { continue }
            address.mSelector = kAudioObjectPropertyName
            var name: CFString = "Built-in microphone" as CFString; bytes = UInt32(MemoryLayout<CFString>.size)
            _ = withUnsafeMutablePointer(to: &name) { pointer in
                AudioObjectGetPropertyData(device, &address, 0, nil, &bytes, pointer)
            }
            return (device, name as String)
        }
        throw SwipeError("No built-in microphone found. Swipe currently requires a MacBook microphone.")
    }
}
