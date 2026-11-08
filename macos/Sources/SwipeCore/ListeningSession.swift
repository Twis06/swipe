/// User intent is separate from temporary OS suspension. Never persist this state.
public struct ListeningSession {
    public enum Blocker: Hashable { case locked, sleeping, displayOff, inactive }
    public private(set) var requested = false
    public private(set) var blockers = Set<Blocker>()
    public private(set) var pendingAction: Blocker?
    public var waitingFor: Set<Blocker> { blockers.union(pendingAction.map { [$0] } ?? []) }
    public var shouldListen: Bool { requested && waitingFor.isEmpty }
    public init() {}
    public mutating func request() { requested = true }
    public mutating func pause() { requested = false; pendingAction = nil }
    public mutating func prepare(_ blocker: Blocker) { pendingAction = blocker }
    public mutating func set(_ blocker: Blocker, present: Bool) {
        if pendingAction == blocker { pendingAction = nil }
        if present { blockers.insert(blocker) } else { blockers.remove(blocker) }
    }
}
