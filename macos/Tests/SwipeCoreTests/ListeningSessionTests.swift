import Testing
import SwipeCore

struct ListeningSessionTests {
    @Test func resumesOnlyAfterAllOSBlockersClear() {
        var session = ListeningSession()
        #expect(!session.shouldListen)
        session.request(); #expect(session.shouldListen)
        session.set(.locked, present: true)
        session.set(.displayOff, present: true)
        session.set(.sleeping, present: true)
        session.set(.sleeping, present: false); #expect(!session.shouldListen)
        session.set(.displayOff, present: false); #expect(!session.shouldListen)
        session.set(.locked, present: false); #expect(session.shouldListen)
        // Duplicate wake/unlock signals don't change intent.
        session.set(.locked, present: false); #expect(session.shouldListen)
    }
    @Test func explicitPauseCancelsResumeAndPendingMappedAction() {
        var session = ListeningSession(); session.request()
        session.prepare(.locked); #expect(!session.shouldListen)
        session.pause(); #expect(session.waitingFor.isEmpty)
        session.set(.locked, present: false); #expect(!session.shouldListen)
        session.request(); #expect(session.shouldListen)
        session.set(.locked, present: true); session.pause()
        session.set(.locked, present: false); #expect(!session.shouldListen)
    }
    @Test func mappedLockBecomesRealLockWithoutBriefResume() {
        var session = ListeningSession(); session.request()
        session.prepare(.locked); #expect(!session.shouldListen)
        session.set(.locked, present: true); #expect(!session.shouldListen)
        #expect(session.pendingAction == nil)
        session.set(.inactive, present: true)
        session.set(.locked, present: false); #expect(!session.shouldListen)
        session.set(.inactive, present: false); #expect(session.shouldListen)
    }
}
