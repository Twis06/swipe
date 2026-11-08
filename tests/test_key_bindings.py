import unittest
from unittest.mock import patch
from swipe.key_bindings import KeyBindings, DEFAULT_BINDINGS, validated_bindings
from swipe import server
from tests.test_recorder import RecorderTest, float_wav


class FakeKeyboard:
    supported = True
    allowed = True
    def __init__(self): self.sent = []
    def permitted(self): return self.allowed
    def request_permission(self): return self.allowed
    def send(self, binding):
        if not self.allowed: raise PermissionError('Permission revoked')
        self.sent.append(binding)


class KeyBindingTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.keyboard = FakeKeyboard()
        self.keys = KeyBindings(self.keyboard, lambda: self.now)
        self.client = 'test-client-123'

    def arm(self): self.keys.configure('arm', self.client, DEFAULT_BINDINGS)
    def offer(self, direction='up', reason='accepted'):
        return self.keys.offer({'prediction': direction, 'reason': reason}, self.client)

    def test_opt_in_and_no_background_or_uncertain_tickets(self):
        self.assertIsNone(self.offer())
        self.arm()
        for direction in ('uncertain', 'background'):
            self.assertIsNone(self.offer(direction))
        self.assertIsNone(self.offer('up', 'unfamiliar_sound'))
        self.assertEqual(self.keyboard.sent, [])

    def test_one_use_cooldown_and_client_ownership(self):
        self.arm(); ticket = self.offer()
        self.assertFalse(self.keys.fire(ticket, 'another-client')['sent'])
        self.assertTrue(self.keys.fire(ticket, self.client)['sent'])
        self.assertFalse(self.keys.fire(ticket, self.client)['sent'])
        self.assertFalse(self.keys.fire(self.offer('down'), self.client)['sent'])
        self.now += .4
        self.assertTrue(self.keys.fire(self.offer('down'), self.client)['sent'])
        self.assertEqual([x['code'] for x in self.keyboard.sent], ['ArrowUp', 'ArrowDown'])

    def test_stop_rearm_and_expiry_invalidate_pending_events(self):
        self.arm(); ticket = self.offer()
        self.keys.configure('stop', self.client)
        self.arm()
        self.assertFalse(self.keys.fire(ticket, self.client)['sent'])
        ticket = self.offer(); self.now += 4
        self.assertFalse(self.keys.fire(ticket, self.client)['sent'])
        ticket = self.offer(); self.now += 21
        self.assertFalse(self.keys.fire(ticket, self.client)['sent'])
        self.assertFalse(self.keys.status(self.client)['enabled'])

    def test_heartbeat_cannot_rearm_expired_session(self):
        self.arm(); self.now = 15; self.keys.configure('heartbeat', self.client)
        self.now = 25; self.assertTrue(self.keys.status(self.client)['enabled'])
        self.now = 40; self.assertFalse(self.keys.configure('heartbeat', self.client)['enabled'])

    def test_permission_and_validation(self):
        self.keyboard.allowed = False
        with self.assertRaises(PermissionError): self.arm()
        self.keyboard.allowed = True; self.arm()
        self.keyboard.allowed = False
        with self.assertRaises(PermissionError): self.keys.fire(self.offer(), self.client)
        self.assertFalse(self.keys.status()['enabled'])
        for invalid in ({}, {'up': {'code': 'shell command', 'modifiers': []}, 'down': DEFAULT_BINDINGS['down']}, {'up': {'code':'KeyA','modifiers':['invalid']}, 'down': DEFAULT_BINDINGS['down']}):
            with self.assertRaises(ValueError): validated_bindings(invalid)

    def test_custom_shortcut_and_bounded_queue(self):
        bindings = {'up': {'code': 'KeyK', 'modifiers': ['meta', 'shift']}, 'down': {'code':'Space','modifiers':[]}}
        self.keys.configure('arm', self.client, bindings)
        for _ in range(100): ticket=self.offer()
        self.assertLessEqual(len(self.keys.tickets),16)
        self.keys.fire(ticket,self.client)
        self.assertEqual(self.keyboard.sent[0], {'code':'KeyK','modifiers':['meta','shift']})


class KeyEndpointTests(RecorderTest):
    def test_ticket_flow_needs_auth_and_model_acceptance(self):
        keyboard=FakeKeyboard(); keys=KeyBindings(keyboard)
        class Preview:
            def classify(self, body): return {'prediction':'up','reason':'accepted'}
        with patch.object(server,'KEYS',keys),patch.object(server,'PREVIEW',Preview()):
            payload={'action':'arm','client':'test-client','bindings':DEFAULT_BINDINGS}
            self.assertEqual(self.request('/api/keys',payload,token='bad')[0],403)
            self.assertEqual(self.request('/api/keys',payload)[0],200)
            # This helper deliberately has no client header: another session gets no ticket.
            _,result=self.request('/api/predict',float_wav([0]*96000))
            self.assertNotIn('action_ticket',result)
            self.assertEqual(self.request('/api/keys/fire',{'ticket':'invented','client':'test-client'})[1]['sent'],False)
            self.assertEqual(keyboard.sent,[])
            self.assertEqual(list(server.DATA.iterdir()),[])
