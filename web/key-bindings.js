// Optional keyboard output. Reloads never arm it automatically.
(() => {
  const $ = id => document.getElementById(id);
  const client = crypto.randomUUID();
  let token, listening = false, armed = false, supported = false, capturing = null, heartbeat, revision = 0, pending = false;
  let bindings = {up: {code: 'ArrowUp', modifiers: []}, down: {code: 'ArrowDown', modifiers: []}};
  let supportedCodes = [];
  const symbols = {meta: '⌘', ctrl: '⌃', alt: '⌥', shift: '⇧'};
  const names = {ArrowUp:'↑', ArrowDown:'↓', ArrowLeft:'←', ArrowRight:'→', Space:'Space', Enter:'Return', Escape:'Esc', Backspace:'Delete', PageUp:'Page Up', PageDown:'Page Down'};
  function label(binding) {
    return ['ctrl','alt','shift','meta'].filter(m => binding.modifiers.includes(m)).map(m => symbols[m]).join('') + (names[binding.code] || binding.code.replace(/^Key|^Digit/, ''));
  }
  function message(text, error = false) { $('keyNotice').textContent = text; $('keyNotice').className = error ? 'error' : 'setup-note'; }
  function paint() {
    for (const direction of ['up', 'down']) {
      const button = $('bind' + (direction === 'up' ? 'Up' : 'Down'));
      button.textContent = capturing === direction ? 'Press a key…' : label(bindings[direction]);
      button.disabled = armed || pending;
      const spoken = bindings[direction].modifiers.map(m => ({meta:'Command',ctrl:'Control',alt:'Option',shift:'Shift'})[m]).concat(names[bindings[direction].code] || bindings[direction].code).join(' plus ');
      button.setAttribute('aria-label', capturing === direction ? 'Press a key or shortcut. Escape cancels.' : `${direction === 'up' ? 'Swipe up' : 'Swipe down'} key: ${spoken}. Click to change.`);
    }
    $('enableKeys').textContent = armed ? 'Disable keys' : 'Enable keys';
    $('enableKeys').disabled = !supported || (!listening && !armed) || Boolean(capturing) || pending;
    $('keyState').textContent = armed ? 'Keys enabled' : 'Keys off';
    $('keyState').dataset.enabled = String(armed);
  }
  async function request(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch(url, {...options, signal: controller.signal});
      const value = await response.json();
      return {ok: response.ok, value};
    }
    finally { clearTimeout(timer); }
  }
  async function post(action, extra = {}) {
    const response = await request('/api/keys', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Recorder-Token': token}, body: JSON.stringify({action, client, ...extra}), keepalive: action === 'stop'});
    const value = response.value;
    if (!response.ok) throw new Error(value.error || 'Key mapping could not connect.');
    return value;
  }
  async function stop() {
    revision++; const wasArmed = armed || pending; armed = false; pending = false; capturing = null; clearInterval(heartbeat); paint();
    if (wasArmed && token) {
      try { await post('stop'); } catch { message('Keys stopped here. The server timeout also disables disconnected sessions.', true); }
    }
  }
  async function toggle() {
    if (armed) { await stop(); message('Key output is off. Recognition can keep running.'); return; }
    if (pending || !listening || !supported) return;
    const attempt = ++revision; pending = true; paint();
    try {
      const result = await post('arm', {bindings});
      if (attempt !== revision || !listening) { await post('stop'); return; }
      armed = result.owned; pending = false; paint();
      if (!armed) return;
      message('Keys go to the app in front. Switch apps and swipe; return here to disable.');
      heartbeat = setInterval(async () => {
        if (!armed) return;
        try {
          const result = await post('heartbeat');
          if (!result.owned) { await stop(); message('Key output timed out. Click Enable keys to resume.', true); }
        } catch { await stop(); message('Key output disconnected. Reconnect before enabling keys.', true); }
      }, 5000);
    } catch (error) { if (attempt === revision) { await stop(); message(error.message, true); } }
    finally { if (attempt === revision) { pending = false; paint(); } }
  }
  function record(direction) {
    if (armed || pending) return;
    capturing = direction; paint(); message('Press your key or shortcut. Escape cancels.');
  }
  document.addEventListener('keydown', event => {
    if (!capturing || armed || pending) return;
    event.preventDefault(); event.stopPropagation();
    if (event.code === 'Escape') { capturing = null; paint(); message('Binding unchanged.'); return; }
    if (['ShiftLeft','ShiftRight','ControlLeft','ControlRight','AltLeft','AltRight','MetaLeft','MetaRight'].includes(event.code)) return;
    if (!supportedCodes.includes(event.code)) { message('Choose a letter, number, arrow, or function key.', true); return; }
    bindings[capturing] = {code: event.code, modifiers: ['meta','ctrl','alt','shift'].filter(m => event[m + 'Key'])};
    capturing = null;
    try { localStorage.setItem('swipe-key-bindings-v1', JSON.stringify(bindings)); } catch {}
    paint(); message('Binding saved. Start listening, then enable key output.');
  }, true);
  $('bindUp').onclick = () => record('up'); $('bindDown').onclick = () => record('down');
  $('enableKeys').onclick = toggle;
  $('keyPermission').onclick = async () => {
    try {
      const status = await post('permission');
      $('keyPermission').hidden = status.permitted;
      message(status.permitted ? 'macOS access is ready. Start listening, then enable keys.' : 'In System Settings → Privacy & Security → Accessibility, allow the app running Swipe (Python, Terminal, or Codex), then return and enable keys.');
    } catch (error) { message(error.message, true); }
  };
  window.swipeKeys = {
    client,
    isArmed: () => armed,
    async configure(value) {
      token = value;
      try {
        const response = await request('/api/keys', {headers: {'X-Swipe-Client': client}});
        const status = response.value;
        if (!response.ok) throw new Error(status.error || 'Key mapping unavailable.');
        supported = status.supported; supportedCodes = status.supported_codes.filter(code => code !== 'Escape');
        try {
          const saved = JSON.parse(localStorage.getItem('swipe-key-bindings-v1'));
          if (saved && ['up','down'].every(d => supportedCodes.includes(saved[d]?.code) && Array.isArray(saved[d]?.modifiers) && saved[d].modifiers.every(m => Object.hasOwn(symbols,m)))) bindings = saved;
        } catch {}
        $('keyPermission').hidden = !supported || status.permitted;
        message(!supported ? 'System key output is available on macOS. Recognition still works here.' : status.permitted ? 'Choose your keys. Start listening, then enable key output.' : 'Allow macOS Accessibility access to send keys. Key output starts off.');
      } catch (error) { message(error.message, true); }
      paint();
    },
    setListening(value) { listening = value; paint(); },
    stop,
    async accepted(result, stillCurrent) {
      if (!armed || !result.action_ticket || !stillCurrent()) return null;
      try {
      const response = await request('/api/keys/fire', {method:'POST', headers:{'Content-Type':'application/json','X-Recorder-Token':token}, body:JSON.stringify({ticket:result.action_ticket,client})});
      const action = response.value;
      if (!response.ok) { await stop(); message(action.error || 'Key output stopped.', true); return null; }
      if (action.sent) return label(action.binding);
      if (action.reason === 'disabled') { await stop(); message('Key output stopped. Enable it again when ready.'); }
      return null;
      } catch (error) { await stop(); message('Key output disconnected. Enable it again when ready.', true); return null; }
    },
  };
  paint();
})();
