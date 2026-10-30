// All computation and file staging go through the worker; no HTTP API exists.
const types = {mrc:'application/octet-stream', png:'image/png', zip:'application/zip', csv:'text/csv', json:'application/json'};
export class ComputeClient extends EventTarget {
  constructor() {
    super();
    this.worker = new Worker(new URL('./worker.js', import.meta.url), {type:'module'});
    this.pending = new Map(); this.assets = new Map(); this.serial = 0;
    this.worker.onmessage = ({data}) => {
      if (data.status) {
        this.dispatchEvent(new CustomEvent('status', {detail:data.status})); return;
      }
      const task = this.pending.get(data.id);
      if (!task) return;
      this.pending.delete(data.id);
      if (data.error) { task.reject(new Error(data.error)); return; }
      for (const file of data.files || []) {
        const old = this.assets.get(file.name);
        if (old) URL.revokeObjectURL(old.url);
        const blob = new Blob([file.buffer], {type:types[file.name.split('.').pop()] || types.mrc});
        this.assets.set(file.name, {url:URL.createObjectURL(blob), buffer:file.name.endsWith('.mrc') ? file.buffer : null});
      }
      task.resolve(data.result);
    };
    this.worker.onerror = event => {
      const error = new Error(event.message || 'The compute worker stopped. Reload to restart the engine.');
      for (const task of this.pending.values()) task.reject(error);
      this.pending.clear();
      this.dispatchEvent(new CustomEvent('status', {detail:error.message}));
    };
  }
  call(method, payload={}, buffer) {
    return new Promise((resolve, reject) => {
      const id = ++this.serial;
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage({id, method, payload, buffer}, buffer ? [buffer] : []);
    });
  }
  name(ref) { return ref.replace(/^asset:/, '').split('?')[0]; }
  url(ref) {
    if (!ref) return '';
    const item = this.assets.get(this.name(ref));
    if (!item) throw new Error('Result file is no longer available. Run this step again.');
    return item.url;
  }
  buffer(ref) { return this.assets.get(this.name(ref))?.buffer; }
  clear(predicate=()=>true) {
    for (const [name, asset] of this.assets) if (predicate(name)) {
      URL.revokeObjectURL(asset.url); this.assets.delete(name);
    }
  }
  dispose() {
    this.worker.terminate(); this.clear();
    for (const task of this.pending.values()) task.reject(new Error('Session reset.'));
    this.pending.clear();
  }
}
