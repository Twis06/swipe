"""Opt-in macOS keystrokes, gated by accepted one-use prediction tickets."""
import ctypes
import secrets
import sys
import threading
import time

# macOS virtual key codes: bindings use physical KeyboardEvent.code names.
KEY_CODES = dict(zip(
    'KeyA KeyS KeyD KeyF KeyH KeyG KeyZ KeyX KeyC KeyV KeyB KeyQ KeyW KeyE KeyR KeyY KeyT Digit1 Digit2 Digit3 Digit4 Digit6 Digit5 Equal Digit9 Digit7 Minus Digit8 Digit0 BracketRight KeyO KeyU BracketLeft KeyI KeyP Enter KeyL KeyJ Quote KeyK Semicolon Backslash Comma Slash KeyN KeyM Period Tab Space Backquote Backspace Escape ArrowLeft ArrowRight ArrowDown ArrowUp Home End PageUp PageDown Delete F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12'.split(),
    [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,123,124,125,126,115,119,116,121,117,122,120,99,118,96,97,98,100,101,109,103,111],
))
MODIFIERS = {'shift': 1 << 17, 'ctrl': 1 << 18, 'alt': 1 << 19, 'meta': 1 << 20}
DEFAULT_BINDINGS = {'up': {'code': 'ArrowUp', 'modifiers': []}, 'down': {'code': 'ArrowDown', 'modifiers': []}}


def validated_bindings(value):
    if not isinstance(value, dict) or set(value) != {'up', 'down'}:
        raise ValueError('Choose one key binding for each direction.')
    result = {}
    for direction, binding in value.items():
        if not isinstance(binding, dict) or set(binding) != {'code', 'modifiers'}:
            raise ValueError('Invalid key binding.')
        code, modifiers = binding['code'], binding['modifiers']
        if not isinstance(code, str) or code not in KEY_CODES:
            raise ValueError('That key is not supported. Choose a letter, number, arrow, or function key.')
        if not isinstance(modifiers, list) or any(not isinstance(m, str) or m not in MODIFIERS for m in modifiers) or len(set(modifiers)) != len(modifiers):
            raise ValueError('Invalid modifier keys.')
        result[direction] = {'code': code, 'modifiers': sorted(modifiers)}
    return result


class MacKeyboard:
    def __init__(self):
        self.supported = sys.platform == 'darwin'
        self.cg = self.cf = None

    def _load(self):
        if self.cg is not None or not self.supported:
            return
        self.cg = ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        self.cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        self.cg.CGPreflightPostEventAccess.restype = ctypes.c_bool
        self.cg.CGRequestPostEventAccess.restype = ctypes.c_bool
        self.cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
        self.cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        self.cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self.cg.CGEventPost.restype = None
        self.cg.CGEventSetFlags.restype = None
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]
        self.cf.CFRelease.restype = None

    def permitted(self):
        self._load()
        return bool(self.supported and self.cg.CGPreflightPostEventAccess())

    def request_permission(self):
        self._load()
        return bool(self.supported and self.cg.CGRequestPostEventAccess())

    def send(self, binding):
        if not self.permitted():
            raise PermissionError('Enable Accessibility access for the app running Swipe, then try again.')
        flags = sum(MODIFIERS[m] for m in binding['modifiers'])
        events = [self.cg.CGEventCreateKeyboardEvent(None, KEY_CODES[binding['code']], down) for down in (True, False)]
        try:
            if not all(events):
                raise RuntimeError('macOS could not create a keyboard event.')
            for event in events:
                self.cg.CGEventSetFlags(event, flags)
                self.cg.CGEventPost(0, event)
        finally:
            for event in events:
                if event:
                    self.cf.CFRelease(event)


class KeyBindings:
    LEASE_SECONDS = 20
    TICKET_SECONDS = 3
    COOLDOWN_SECONDS = .35

    def __init__(self, keyboard=None, clock=time.monotonic):
        self.keyboard = keyboard or MacKeyboard()
        self.clock = clock
        self.lock = threading.RLock()
        self.bindings = validated_bindings(DEFAULT_BINDINGS)
        self.owner = None
        self.deadline = 0
        self.tickets = {}
        self.last_sent = float('-inf')

    def _expire(self):
        if self.owner and self.clock() >= self.deadline:
            self.owner = None
            self.tickets.clear()

    def status(self, client=''):
        with self.lock:
            self._expire()
            return {'supported': self.keyboard.supported, 'permitted': self.keyboard.permitted(),
                    'enabled': self.owner is not None, 'owned': bool(client and client == self.owner),
                    'bindings': self.bindings, 'supported_codes': list(KEY_CODES)}

    def configure(self, action, client, bindings=None):
        if not isinstance(client, str) or not 8 <= len(client) <= 100:
            raise ValueError('Reload the preview to reconnect key mapping.')
        with self.lock:
            self._expire()
            if action == 'permission':
                self.keyboard.request_permission()
            elif action == 'arm':
                chosen = validated_bindings(bindings)
                if not self.keyboard.supported:
                    raise ValueError('System key mapping currently requires macOS.')
                if not self.keyboard.permitted():
                    raise PermissionError('Enable Accessibility access for the app running Swipe in System Settings, then click Enable keys again.')
                if self.owner and self.owner != client:
                    raise ValueError('Another preview is sending keys. Stop it first.')
                self.bindings, self.owner = chosen, client
                self.deadline = self.clock() + self.LEASE_SECONDS
                self.tickets.clear()
            elif action == 'heartbeat':
                if self.owner == client:
                    self.deadline = self.clock() + self.LEASE_SECONDS
            elif action == 'stop':
                if self.owner == client:
                    self.owner = None
                    self.tickets.clear()
            else:
                raise ValueError('Unknown key-mapping action.')
            return self.status(client)

    def offer(self, prediction, client):
        with self.lock:
            self._expire()
            if self.owner != client or not self.owner or prediction.get('prediction') not in ('up', 'down') or prediction.get('reason') != 'accepted':
                return None
            now = self.clock()
            self.tickets = {k: v for k, v in self.tickets.items() if v[0] > now}
            if len(self.tickets) >= 16:
                self.tickets.pop(next(iter(self.tickets)))
            ticket = secrets.token_urlsafe(24)
            self.tickets[ticket] = (now + self.TICKET_SECONDS, prediction['prediction'])
            return ticket

    def fire(self, ticket, client):
        if not isinstance(ticket, str):
            raise ValueError('Invalid prediction ticket.')
        with self.lock:
            self._expire()
            if not self.owner or self.owner != client:
                return {'sent': False, 'reason': 'disabled'}
            value = self.tickets.pop(ticket, None)
            now = self.clock()
            if value is None or value[0] <= now:
                return {'sent': False, 'reason': 'expired_or_used'}
            if now - self.last_sent < self.COOLDOWN_SECONDS:
                return {'sent': False, 'reason': 'cooldown'}
            try:
                self.keyboard.send(self.bindings[value[1]])
            except (PermissionError, RuntimeError):
                self.owner = None
                self.tickets.clear()
                raise
            self.last_sent = now
            return {'sent': True, 'direction': value[1], 'binding': self.bindings[value[1]]}
