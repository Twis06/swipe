import Combine

/// Meter ticks invalidate only the meter view, not the settings form or menu bar.
@MainActor final class AudioMeter: ObservableObject {
    @Published var value: Double = 0
}
