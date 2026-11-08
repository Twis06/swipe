import CoreAudio
import AudioToolbox
import SwipeCore

final class Microphone {
    private var audioQueue: AudioQueueRef?
    private let queue = DispatchQueue(label: "swipe.capture", qos: .userInitiated)
    private let rate = 48000
    private var capture: GestureCapture?
    private var pending: [Float] = [], frames = 0, peak: Float = 0, levelFrame = 0
    private var epoch = 0
    private var watchdog: DispatchSourceTimer?
    var onCandidate: ((Candidate, Int) -> Void)?
    var onLevel: ((Float, Bool) -> Void)?
    var onFailure: ((String) -> Void)?

    func start() throws -> String {
        let device = try Self.builtInInput()
        let current = queue.sync { () -> Int in
            epoch += 1; capture = GestureCapture(rate: rate)
            pending = []; frames = 0; peak = 0; levelFrame = 0
            capture?.onCandidate = { [weak self] candidate in
                guard let self else { return }; self.onCandidate?(candidate, self.rate)
            }
            return epoch
        }
        var format = AudioStreamBasicDescription(mSampleRate: Double(rate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
            mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        var created: AudioQueueRef?
        let creation = AudioQueueNewInputWithDispatchQueue(&created, &format, 0, queue) { [weak self] input, buffer, _, _, _ in
            guard let self, self.epoch == current else { return }
            let size = Int(buffer.pointee.mAudioDataByteSize)
            guard size % MemoryLayout<Float>.size == 0 else {
                self.onFailure?("The microphone returned an invalid audio buffer. Start listening again."); return
            }
            let samples = buffer.pointee.mAudioData.assumingMemoryBound(to: Float.self)
            let pcm = Array(UnsafeBufferPointer(start: samples, count: size / MemoryLayout<Float>.size))
            // Return the buffer before feature capture work; only copied samples leave this callback.
            let status = AudioQueueEnqueueBuffer(input, buffer, 0, nil)
            guard status == noErr else { self.onFailure?("Microphone capture stopped. Start listening again."); return }
            self.consume(pcm)
        }
        guard creation == noErr, let input = created else { throw SwipeError("Could not create microphone capture (macOS \(creation)).") }
        audioQueue = input
        do {
            // An input-only queue can pin its device without rerouting the system input/output.
            var uid = device.uid
            let selection = withUnsafePointer(to: &uid) { pointer in
                AudioQueueSetProperty(input, kAudioQueueProperty_CurrentDevice, pointer, UInt32(MemoryLayout<CFString>.size))
            }
            guard selection == noErr else { throw SwipeError("Could not select the built-in microphone (macOS \(selection)).") }
            for _ in 0..<3 {
                var buffer: AudioQueueBufferRef?
                guard AudioQueueAllocateBuffer(input, 1024 * 4, &buffer) == noErr, let buffer,
                      AudioQueueEnqueueBuffer(input, buffer, 0, nil) == noErr else { throw SwipeError("Could not prepare microphone buffers.") }
            }
            let started = AudioQueueStart(input, nil)
            guard started == noErr else { throw SwipeError("Could not start the built-in microphone (macOS \(started)).") }
            queue.sync {
                var progress = CaptureProgress(now: ProcessInfo.processInfo.systemUptime)
                let timer = DispatchSource.makeTimerSource(queue: queue)
                timer.schedule(deadline: .now() + 1, repeating: 1)
                timer.setEventHandler { [weak self] in
                    guard let self, self.epoch == current else { return }
                    if progress.isStalled(frames: self.frames, now: ProcessInfo.processInfo.systemUptime) {
                        self.watchdog?.cancel()
                        self.onFailure?("The microphone stopped delivering audio. Start listening again to reconnect.")
                    }
                }
                watchdog = timer; timer.resume()
            }
        } catch { stop(); throw error }
        return device.name
    }

    // Runs only on the capture queue, at the same 128-frame boundaries as the browser worklet.
    private func consume(_ pcm: [Float]) {
        guard pcm.allSatisfy({ $0.isFinite }) else { onFailure?("The microphone returned invalid audio. Start listening again."); return }
        pending.append(contentsOf: pcm)
        var offset = 0
        while pending.count - offset >= 128 {
            let block = Array(pending[offset..<(offset + 128)])
            capture?.process(block); frames += 128
            peak = max(peak, block.map { abs($0) }.max() ?? 0); offset += 128
            if frames - levelFrame >= rate / 10 {
                onLevel?(peak, frames < rate); peak = 0; levelFrame = frames
            }
        }
        if offset > 0 { pending.removeFirst(offset) }
    }
    func accept(_ id: Int) { queue.async { [weak self] in self?.capture?.accept(id) } }
    func stop() {
        let old = audioQueue; audioQueue = nil
        // Synchronous disposal finishes callbacks before their state is reset or released.
        if let old { AudioQueueDispose(old, true) }
        queue.sync {
            epoch += 1; watchdog?.cancel(); watchdog = nil
            capture?.stop(); capture = nil; pending = []; peak = 0
        }
    }
    static func builtInInput() throws -> (id: AudioDeviceID, name: String, uid: CFString) {
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
            address.mSelector = kAudioDevicePropertyDeviceUID
            var uid: CFString = "" as CFString; bytes = UInt32(MemoryLayout<CFString>.size)
            let uidStatus = withUnsafeMutablePointer(to: &uid) { pointer in
                AudioObjectGetPropertyData(device, &address, 0, nil, &bytes, pointer)
            }
            guard uidStatus == noErr, !(uid as String).isEmpty else { continue }
            return (device, name as String, uid)
        }
        throw SwipeError("No built-in microphone found. Swipe currently requires a MacBook microphone.")
    }
}
