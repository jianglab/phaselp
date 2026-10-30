// Single-threaded WASM: works on static hosting without COOP/COEP or SharedArrayBuffer.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.mjs';

const WHEEL = 'vendor/phaselp-1.1.1-py3-none-any.whl';
const status = message => self.postMessage({ status: message });
let engine, plotting;
async function initialize() {
  status('Loading compute engine… the first visit downloads the scientific runtime.');
  const py = await loadPyodide();
  status('Loading NumPy and SciPy…');
  await py.loadPackage(['numpy', 'scipy', 'micropip']);
  py.globals.set('_site_url', new URL('./', import.meta.url).href);
  await py.runPythonAsync(`
import micropip
await micropip.install(_site_url + 'vendor/mrcfile-1.5.4-py2.py3-none-any.whl', deps=False)
await micropip.install(_site_url + '${WHEEL}', deps=False)
from phaselp.webapp import session
session.configure(site_url=_site_url)
`);
  status('Compute engine ready · files stay in this browser');
  return py;
}

// matplotlib (about 10 MB) is needed only for the step figures, so it loads after the input maps.
const FIGURES = new Set(['warm', 'register', 'filter', 'amplitude', 'compare']);
function loadPlotting(py) {
  return plotting ||= py.loadPackage(['matplotlib'])
    .then(() => py.runPythonAsync("import matplotlib\nmatplotlib.use('Agg')\nfrom phaselp.webapp import figures"))
    .catch(error => { plotting = null; throw error; });
}

async function handle({ id, method, payload = {}, buffer }) {
  try {
    const py = await (engine ||= initialize());
    if (FIGURES.has(method)) await loadPlotting(py);
    if (method === 'warm') { self.postMessage({ id, result: { ok: true } }); return; }
    if (method === 'proof') {
      const result = await py.runPythonAsync(`
import json, numpy as np, scipy, mrcfile, phaselp
_tiny = np.random.default_rng(7).normal(size=(12, 12, 12))
_fsc = phaselp.fsc(_tiny, _tiny)[1]
json.dumps({'ok': bool(np.allclose(_fsc[np.isfinite(_fsc)], 1)), 'phaselp': phaselp.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'mrcfile': mrcfile.__version__})
`);
      self.postMessage({ id, result: JSON.parse(result) });
      return;
    }
    py.globals.set('_method', method);
    py.globals.set('_payload_json', JSON.stringify(payload));
    py.globals.set('_incoming', buffer ? new Uint8Array(buffer) : null);
    const result = JSON.parse(await py.runPythonAsync(
      'await session.dispatch_json(_method, _payload_json, _incoming)'));
    const files = [], transfers = [];
    for (const name of result._assets || []) {
      py.globals.set('_file_name', name);
      const proxy = py.runPython('session.file_bytes(_file_name)');
      try {
        const copy = new Uint8Array(proxy.toJs());
        files.push({ name, buffer: copy.buffer });
        transfers.push(copy.buffer);
      } finally { proxy.destroy(); }
    }
    delete result._assets;
    self.postMessage({ id, result, files }, transfers);
    py.runPython('_incoming = None');
  } catch (error) {
    self.postMessage({ id, error: String(error.message || error) });
  }
}
// A queue is essential: async fetches must not interleave separate Python jobs.
let queue = Promise.resolve();
self.onmessage = event => { queue = queue.then(() => handle(event.data)); };
