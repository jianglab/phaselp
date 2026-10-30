import {ComputeClient} from './client.js';
import {parseViewer, drawSlice} from './slice.js';
import {enhance, inspect, attachPlot} from './hover.js';
const compute = new ComputeClient();
window.phaselp = compute;
compute.addEventListener('status', e => document.querySelector('#engineStatus').textContent=e.detail);
const escapeHTML = value => String(value ?? '—').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const errorMessage = error => String(error.message||error).trim().split('\n').filter(Boolean).pop().replace(/^(ValueError|RuntimeError|Error): /,'');
let BUSY = false;
function busy(on) {
  BUSY=on;
  document.querySelectorAll('main button, main input, main select').forEach(el=>{
    if(el.id==='resetBtn') return;
    if(on) { el.dataset.wasDisabled=el.disabled?'1':'0'; el.disabled=true; }
    else { el.disabled=el.dataset.wasDisabled==='1'; delete el.dataset.wasDisabled; }
  });
}
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
function withSid(url){ return compute.url(url); }

// source-type sub-input toggling
$$(".src").forEach(src => {
  const sel = $(".srcType", src);
  const upd = () => {
    $(".srcUpload", src).hidden = sel.value !== "upload";
    $(".srcEmd", src).hidden = sel.value !== "emd";
    $(".srcPdb", src).hidden = sel.value !== "pdb";
  };
  sel.addEventListener("change", upd); upd();
});

// optional low-resolution reference: when off, hide the low map source and the
// alignment options (registration only makes sense against a reference)
const noRefBox = $("#noRef");
function updRefVisibility(){
  const off = noRefBox.checked;
  $('.src[data-role="low"] .lowbody').hidden = off;   // hide the inputs, keep the checkbox
  $("#alignParams").hidden = off;
}
noRefBox.addEventListener("change", updRefVisibility); updRefVisibility();

async function readSource(role){
  const src = $(`.src[data-role="${role}"]`);
  const type = $(".srcType", src).value;
  if(type === "demo") return {type:"demo"};
  if(type === "upload"){
    const f = $(".srcUpload input", src).files[0];
    if(!f) throw new Error(`choose a file for the ${role}-resolution map`);
    const buf = await f.arrayBuffer();
    const j = await compute.call("upload", {name:f.name}, buf);
    return {type:"upload", filename:j.filename};
  }
  if(type === "emd"){
    const v = $(".srcEmd input", src).value.trim();
    if(!v) throw new Error(`enter an EMD id for the ${role}-resolution map`);
    return {type:"emd", value:v};
  }
  const v = $(".srcPdb input", src).value.trim();
  if(!v) throw new Error(`enter a PDB id for the ${role}-resolution map`);
  return {type:"pdb", value:v,
          sim_res:parseFloat($(".simres", src).value), sim_apix:parseFloat($(".simapix", src).value)};
}

// ---- 3-D isosurface viewer (rotation-syncable) -------------------------
let VUID = 0;
class Vol {
  constructor(host){
    this.stage = new NGL.Stage(host, {backgroundColor:"#0b0d10"});
    this.comp = null; this.surf = null; this._iso = 1.5;
    this.disposed = false; this.listeners = []; this.loading = Promise.resolve();
  }
  _surface(){
    return this.comp.addRepresentation("surface",
      {isolevelType:"sigma", isolevel:this._iso, colorScheme:"uniform", colorValue:0x7fb8e6,
       opacity:1.0, side:"double"});
  }
  async load(url){
    this.loading = this.stage.loadFile(withSid(url), {ext:"mrc"});
    const component = await this.loading;
    if(this.disposed) return;
    this.comp = component;
    this.surf = this._surface();
    await this.waitIdle();
    // recenter is done by the caller once the container has its final size
  }
  // Swap in a new map on the SAME stage, keeping the current camera, so
  // changing filter mode/resolution updates the density in place instead of
  // resetting the viewport.
  async reload(url){
    const keep = this.stage.viewerControls.getOrientation();
    await this.loading.catch(()=>{}); await this.waitIdle();
    if(this.disposed) return;
    if(this.comp){ try{ this.stage.removeComponent(this.comp); }catch(e){} this.comp = null; this.surf = null; }
    await this.load(url);
    if(this.disposed) return;
    try{ this.stage.viewerControls.orient(keep); }catch(e){}   // restore the viewport
  }
  // Fit the volume to the viewport, then pull back a little so the density
  // sits inside the frame with margin rather than filling it edge-to-edge.
  // Must run after the canvas has its real size.
  recenter(){
    if(this.disposed) return;
    try{
      this.stage.handleResize();
      if(!this.comp) return;
      this.comp.autoView(0);   // instant fit — an animated fit leaves getOrientation() reading a stale camera
      const c = this.stage.viewerControls, m = c.getOrientation();
      // scale the view matrix's basis (== camera distance) up to pull back ~55%
      if(m && m.elements){ const f = 1.55; [0,1,2,4,5,6,8,9,10].forEach(i => m.elements[i] *= f); c.orient(m); }
    }catch(e){}
  }
  setIso(v){ this._iso = parseFloat(v); if(this.surf && !this.disposed) this.surf.setParameters({isolevel:this._iso}); }
  getOrientation(){ return this.stage.viewerControls.getOrientation(); }
  setOrientation(m){ if(this.disposed) return; try{ this.stage.viewerControls.orient(m); }catch(e){} }
  onChange(cb){
    if(this.disposed) return ()=>{};
    const s=this.stage.viewerControls.signals.changed;
    const guarded=()=>{if(!this.disposed) cb();};
    s.add(guarded); const off=()=>s.remove(guarded); this.listeners.push(off); return off;
  }
  waitIdle(){ return new Promise(resolve=>this.stage.tasks.onZeroOnce(resolve)); }
  dispose(){
    if(this.disposed) return;
    this.disposed=true;
    this.listeners.forEach(off=>off()); this.listeners=[];
    // NGL's surface workers and file loaders can still dispatch signals. Wait
    // for them before disposing their counters (particularly visible in WebKit).
    this.loading.catch(()=>{}).then(()=>this.waitIdle()).then(()=>setTimeout(()=>this.stage.dispose(),0));
  }
  resize(){ if(this.disposed) return; try{ this.stage.handleResize(); }catch(e){} }
  // 3x3 rotation (row-major) from the model->screen orientation matrix
  orient9(){
    const e = this.stage.viewerControls.getOrientation().elements;
    return [e[0],e[4],e[8], e[1],e[5],e[9], e[2],e[6],e[10]];
  }
}
const VIEWERS = {};
window.addEventListener("resize", () => Object.values(VIEWERS).forEach(v => { try{ v.stage.handleResize(); }catch(e){} }));
// dispose NGL stages inside an element before its content is replaced (frees WebGL contexts)
function disposeViewersIn(el){
  if(!el) return;
  el.querySelectorAll(".viewer").forEach(host => {
    const v = VIEWERS[host.id];
    if(v && v.stage) v.dispose();
    delete VIEWERS[host.id];
  });
}

// a viewer box = 3-D isosurface | live 2-D section, with iso + slice controls
function mountViewer(container, url, label){
  const id = "vw" + (++VUID);
  let currentUrl = url, sliceVolume = null;
  const box = document.createElement("div"); box.className = "vbox";
  box.innerHTML = `
    <div class="vwrap">
      <div class="viewer" id="${id}"><span class="cap">${escapeHTML(label||"")}</span><span class="inspect-caption">Drag to rotate · scroll to zoom</span></div>
      <canvas class="slice2d" id="${id}img" aria-label="Live density section; hover to inspect" role="img"></canvas>
    </div>
    <div class="vctl">
      <span class="grp">iso <input type="range" class="iso" min="0.5" max="6" step="0.1" value="1.5" aria-label="Isosurface threshold"><output class="isoValue">1.5σ</output></span>
      <span class="grp">depth <input type="range" class="depth" min="0" max="100" step="1" value="50" aria-label="Section depth"><output class="depthValue">50%</output></span>
    </div>`;
  container.appendChild(box);
  const host = $("#"+id, box); const img = $("#"+id+"img", box);
  let vol = null;
  const refreshImg = () => {
    if(!box.isConnected) return;
    const depth = $('.depth',box).value;
    $('.depthValue',box).textContent=depth+'%';
    $('.isoValue',box).textContent=Number($('.iso',box).value).toFixed(1)+'σ';
    try {
      sliceVolume ||= parseViewer(compute.buffer(currentUrl));
      drawSlice(img,sliceVolume,vol?.comp ? vol.orient9() : null,Number(depth));
    } catch(e) { img.setAttribute('aria-label', 'Slice unavailable: '+errorMessage(e)); }
  };
  // throttle: at most ~6 updates/s during a drag, plus a trailing refresh
  let last = 0, tId = null;
  const schedule = () => {
    const now = Date.now(); clearTimeout(tId);
    if(now - last > 160){ last = now; refreshImg(); }
    else tId = setTimeout(() => { last = Date.now(); refreshImg(); }, 160);
  };
  try { vol = new Vol(host); VIEWERS[id] = vol; requestAnimationFrame(() => vol.resize()); }
  catch(e){ host.innerHTML = `<span class="cap">3-D viewer unavailable</span>`; }
  let ready = Promise.resolve(false);
  if(vol){
    ready = vol.load(url)
       .then(() => new Promise(res => requestAnimationFrame(() => {   // let the container reach its final size first
         if(vol.disposed || !box.isConnected){res(false);return;}
         vol.recenter(); refreshImg(); vol.onChange(schedule); res(true);
       })))
       .catch(e => { host.innerHTML = `<span class="cap">3-D view failed (${e})</span>`; refreshImg(); return false; });
  } else {
    refreshImg();
  }
  $(".iso", box).addEventListener("input", e => { if(vol) vol.setIso(e.target.value); refreshImg(); });
  enhance(box);
  host.addEventListener("pointermove", e=>{if(e.buttons){inspect.hide();return;} if(!sliceVolume)return; const v=sliceVolume; inspect.show(e,label||"Density isosurface", [`Threshold ${vol?vol._iso:1.5}σ · density ${(v.mean+(vol?vol._iso:1.5)*v.sigma).toPrecision(4)}`,`Viewer grid ${v.nx} × ${v.ny} × ${v.nz} · ${v.voxel.toFixed(3)} Å/voxel`,"Drag to rotate · scroll to zoom · hover the slice for point values"]);});
  host.addEventListener("pointerleave",inspect.hide);
  $(".depth", box).addEventListener("input", schedule);
  return {
    vol, box, ready,
    setLabel(value){ label=value; const caption=$('.cap',host); if(caption) caption.textContent=value; },
    setIso(v){ $(".iso", box).value = v; if(vol) vol.setIso(v); refreshImg(); },
    setDepth(v){ $(".depth", box).value = v; refreshImg(); },
    reload(u){ currentUrl=u; sliceVolume=null; if(!vol){refreshImg();return Promise.resolve(false);}
      return vol.reload(u).then(() => { refreshImg(); return true; })
                          .catch(() => { refreshImg(); return false; }); },
  };
}

// a pair of viewers with an optional rotation-sync toggle
function mountPair(container, items){
  const wrap = document.createElement("div");
  const sync = document.createElement("div"); sync.className = "pairsync";
  sync.innerHTML = `<label><input type="checkbox" class="synced" checked> Sync</label>`;
  const grid = document.createElement("div"); grid.className = "viewpair";
  wrap.appendChild(sync); wrap.appendChild(grid); container.appendChild(wrap);
  const handles = items.map(it => mountViewer(grid, it.url, it.label));
  const box = $(".synced", sync);
  if(handles.length === 2 && handles[0].vol && handles[1].vol){
    const [ha, hb] = handles; const a = ha.vol, b = hb.vol; let guard = false;
    const enabled = () => box.checked;
    const mirror = (src, dst) => () => {
      if(guard || !enabled()) return;
      guard = true; dst.setOrientation(src.getOrientation());
      requestAnimationFrame(() => { guard = false; });
    };
    a.onChange(mirror(a, b)); b.onChange(mirror(b, a));
    // Sync also couples the iso and depth sliders (setIso/setDepth don't
    // re-dispatch input, so there is no feedback loop).
    $(".iso", ha.box).addEventListener("input", e => { if(enabled()) hb.setIso(e.target.value); });
    $(".iso", hb.box).addEventListener("input", e => { if(enabled()) ha.setIso(e.target.value); });
    $(".depth", ha.box).addEventListener("input", e => { if(enabled()) hb.setDepth(e.target.value); });
    $(".depth", hb.box).addEventListener("input", e => { if(enabled()) ha.setDepth(e.target.value); });
    const align = () => { b.setOrientation(a.getOrientation());
      hb.setIso($(".iso", ha.box).value); hb.setDepth($(".depth", ha.box).value); };
    box.addEventListener("change", () => { if(box.checked) align(); });
    // Sync is on by default: align the two views once both maps have loaded.
    Promise.all([ha.ready, hb.ready]).then(ready => { if(ready.every(Boolean) && box.checked) align(); });
  } else {
    sync.style.display = "none";
  }
  enhance(wrap);
  return {handles, wrap, sync: box};
}

// ---- Load inputs ----
$("#loadBtn").addEventListener("click", async () => {
  const btn = $("#loadBtn"); const st = $("#loadStatus"); const err = $("#inputErr");
  if(BUSY) return;
  busy(true);
  err.hidden = true; btn.disabled = true;
  disposeViewersIn($("#inputPair")); disposeViewersIn($("#steps"));
  $("#inputPair").innerHTML=""; $("#steps").hidden=true; $("#inputResult").hidden=true;
  compute.clear(); LAST_FSC=null;
  st.innerHTML = '<span class="spinner"></span> loading maps…';
  try{
    const useRef = !noRefBox.checked;
    const high = await readSource("high");
    const payload = {high,
      orientation_search:$("#orient").value,
      translation_bound:parseFloat($("#transBound").value||"20"),
      max_dimension:parseInt($("#maxDim").value||"64"),
      align_res:Number($("#alignRes").value), emdb_max_dimension:Number($("#emdbCap").value),
      standardize:$("#standardize").checked,
      skip_register:$("#skipReg").checked};
    payload.low = useRef ? await readSource("low") : {type:"none"};
    if(high.type === "demo") st.innerHTML = '<span class="spinner"></span> downloading the demo maps (5 MB)…';
    const j = await compute.call("input", payload);
    HAS_REF = !!j.has_reference;
    INPUT_VOXEL = HAS_REF ? j.low.voxel : j.high.voxel;
    HIGH_DIM = Math.max(...j.high.shape);
    const meta = m => `${escapeHTML(m.source)} · ${m.shape.join("×")} · ${m.voxel} Å/vox` +
        (m.reported_resolution_A? ` · ${m.reported_resolution_A} Å` : "") +
        (m.n_atoms? ` · ${m.n_atoms} atoms` : "") + (m.title? `<br><span class="hint">${escapeHTML(m.title)}</span>`:"") + (m.note? `<br><span class="hint">${escapeHTML(m.note)}</span>`:"");
    $("#highMeta").innerHTML = meta(j.high);
    $("#lowMeta").innerHTML = (HAS_REF && j.low) ? meta(j.low)
        : '<span class="muted">no low-res reference — envelope filter only</span>';
    disposeViewersIn($("#inputPair")); $("#inputPair").innerHTML = "";
    $("#inputResult").hidden = false;   // reveal before mounting so NGL sizes correctly
    const items = [{url:j.high_viewer, label:"high-res input"}];
    if(HAS_REF && j.low_viewer) items.push({url:j.low_viewer, label:"low-res input"});
    mountPair($("#inputPair"), items);
    initSteps();
    st.innerHTML = '<span style="color:var(--ok)">✓ loaded — run the steps below</span>';
    compute.call("warm").catch(() => {});   // plotting tools load while you look at the maps
  }catch(e){ err.hidden=false; err.textContent = errorMessage(e); st.textContent=""; }
  finally{ busy(false); btn.disabled=false; if(!$("#steps").hidden) updateEnable(); enhance(); }
});
$("#resetBtn").addEventListener("click", () => {
  compute.dispose();
  const next = new URL(location.href);
  next.searchParams.set('reset','1'); next.searchParams.delete('no-demo');
  location.replace(next.href);
});
// Inputs panel is collapsible too (setCollapsed is hoisted, defined below)
$("#inputHead").addEventListener("click", () =>
  setCollapsed($("#inputPanel"), !$("#inputPanel").classList.contains("collapsed")));

// ---- Steps ----
// The pipeline is dynamic: which steps exist depends on whether a reference was
// given and which post-alignment mode is chosen.
let HAS_REF = true;      // set from the load response
let INPUT_VOXEL = 1;
let HIGH_DIM = 0;        // longest axis of the high map, for the no-reference resample default
let MODE = "ramped";     // ramped (phase + amplitude) | amplitude (amplitude only) | cutoff (hard cutoff)
let STEP_ORDER = ["register","filter","amplitude","compare"];

function activeSteps(){
  if(!HAS_REF) return ["register","filter"];           // resampling is optional without a reference
  // every mode can be matched to the reference's radial amplitude profile (Step 3)
  return ["register","filter","amplitude","compare"];
}
function configureSteps(){
  STEP_ORDER = activeSteps();
  ["register","filter","amplitude","compare"].forEach(n =>
    $("#step-"+n).hidden = !STEP_ORDER.includes(n));
}
let PRODUCED = {aligned:false, filtered:false, matched:false};  // which pipeline products currently exist
function initSteps(){
  $("#steps").hidden = false;
  MODE = "ramped"; LAST_FSC=null; FILTER_VIEWS=null; FILTER_ACTIVE=false;
  ["register","filter","amplitude","compare"].forEach(n=>{const el=$("#step-"+n);disposeViewersIn(el);el.innerHTML="";});
  PRODUCED = {aligned:false, filtered:false, matched:false};
  configureSteps();
  STEP_ORDER.forEach((name, i) => renderStepShell(name, i===0 || (!HAS_REF && name==='filter')));
  updateEnable();
}
// Filter and amplitude matching are optional. A step's Run is enabled whenever
// its prerequisite product exists, so the user can jump straight to Compare
// (a direct high-vs-low comparison) without filtering.
function enableStep(name){
  const el = $("#step-"+name); if(!el || el.hidden) return;
  el.classList.add("active");
  const b = $("[data-run]", el); if(b) b.disabled = false;
  if(name === "filter") setFilterEnabled(true);
}
function updateEnable(){
  const available={register:true,filter:!HAS_REF||PRODUCED.aligned,amplitude:HAS_REF&&PRODUCED.filtered,compare:HAS_REF&&PRODUCED.aligned};
  for(const name of STEP_ORDER){
    const el=$('#step-'+name), run=$('[data-run]',el), on=!!available[name]&&!BUSY;
    if(run) run.disabled=!on;
    el.classList.toggle('active',!!available[name]);
  }
  setFilterEnabled(!!available.filter&&!BUSY);
}

let LAST_FSC = null;      // most recent filter fsc_chart, so "Construct to match" can read the parent–low 0.5 crossing
let FILTER_ACTIVE = false;
let FILTER_VIEWS = null;  // persistent filter viewer pair, reloaded (not rebuilt) across re-runs to keep the camera

function filterControlsHTML(){
  return `
    <div class="filterctl">
      <div class="filterbar">
        <span class="filterto"><label>Filter to</label>
          <input type="range" id="fResRange" min="2" max="60" step="0.5" value="10">
          <input type="number" id="fTargetRes" min="2" step="0.1" value="10"><span class="unit">Å</span></span>
        <span class="modes" role="radiogroup">
          <label><input type="radio" name="fmode" value="ramped" checked> Ramped phase</label>
          <label><input type="radio" name="fmode" value="amplitude"> Amplitude only</label>
          <label><input type="radio" name="fmode" value="cutoff"> Hard cutoff</label>
        </span>
      </div>
      <div class="params" style="margin-top:10px">
        <div data-need="env"><label for="fParentRes">parent resolution (Å)</label><input type="number" id="fParentRes" value="0" min="0" step="0.1"><div class="hint">eq. 7 correction; 0 = ideal parent (eq. 5)</div></div>
        <div data-need="env"><label for="fKappa">κ (SSNR scale)</label><input type="number" id="fKappa" value="100" min="1" step="1"><div class="hint">eq. 5; 100 in the manuscript</div></div>
        <div data-need="seeds"><label for="fSeeds">seeds</label><input type="number" id="fSeeds" value="4" min="1" max="32" step="1"></div>
      </div>
      <div class="hint">Changing a filter parameter recalculates the map when you commit the value (Enter or leave the field).</div>
    </div>
    <div class="fscacts" data-fscacts>
      <button data-act="construct">Construct to match</button>
      <button data-act="trace">Trace profile</button>
    </div>
    ${HAS_REF?'':'<div class="hint">Construct to match and Trace profile require a reference map.</div>'}
    <div data-result></div>
    <div data-viewers></div>`;
}
// build the filter viewer pair once, then reload the filtered map in place on
// later runs so the camera (viewport) is preserved
function renderFilterViewers(container, items){
  const reusable = FILTER_VIEWS && FILTER_VIEWS.container === container
      && document.body.contains(container)
      && FILTER_VIEWS.handles.length === items.length
      && items.length >= 1;
  if(reusable){
    FILTER_VIEWS.handles[0].setLabel(items[0].label);
    FILTER_VIEWS.handles[0].reload(items[0].url);   // only the filtered map changes; the reference is unchanged
    return;
  }
  disposeViewersIn(container); container.innerHTML = "";
  const pair = mountPair(container, items);
  FILTER_VIEWS = {container, handles: pair.handles};
}
function refreshActs(el){
  el = el || $("#step-filter");
  const acts = $(".fscacts", el); if(!acts) return;
  $$("button", acts).forEach(b => {
    b.disabled = !HAS_REF || !FILTER_ACTIVE || MODE !== "ramped" || (b.dataset.act==="construct" && !LAST_FSC);
    b.title = HAS_REF ? '' : 'Load a reference map to match its FSC.';
  });
}
function applyModeVisibility(el){
  $$('input[name="fmode"]', el).forEach(r => { r.checked = (r.value === MODE); });
  $$("[data-need]", el).forEach(d => {
    const n = d.dataset.need;
    d.hidden = (n==="env" && MODE==="cutoff") || (n==="seeds" && MODE!=="ramped");
  });
  refreshActs(el);
}
function setFilterEnabled(on){
  FILTER_ACTIVE = on;
  const el = $("#step-filter");
  $$('.filterctl input', el).forEach(input => input.disabled = !on);
  refreshActs(el);
}
function setFilterRes(r){   // keep the slider and number box in sync
  const el = $("#step-filter"); const num = $("#fTargetRes", el), rng = $("#fResRange", el);
  if(!num) return;
  r = Math.max(2, r); num.value = Math.round(r*10)/10;
  if(rng) rng.value = Math.min(parseFloat(rng.max), Math.max(parseFloat(rng.min), r));
}
function setFilterGrid(voxel){
  const nyquist = Math.ceil(2 * voxel * 10) / 10;
  $('#fResRange').min = nyquist; $('#fTargetRes').min = nyquist;
  $('#fResRange').max = Math.max(60, nyquist * 2);
  setFilterRes(Math.max(Number($('#fTargetRes').value)||10, nyquist));
}
function crossHalf(curve){   // first downward 0.5 crossing of an FSC curve {freq,fsc}
  if(!curve) return null;
  const f = curve.freq||[], v = curve.fsc||[];
  for(let i=1;i<f.length;i++){ const a=v[i-1], b=v[i];
    if(isFinite(a)&&isFinite(b)&&a>=0.5&&b<0.5){ const t=(0.5-a)/(b-a); return f[i-1]+t*(f[i]-f[i-1]); } }
  return null;
}
function handleFilterAct(act){   // every toolbar button runs the filter
  if(BUSY || !FILTER_ACTIVE || (!HAS_REF && ['construct','trace'].includes(act))) return;
  let extra = {};
  if(act==="construct"){
    MODE = "ramped";
    const s = crossHalf(LAST_FSC && LAST_FSC.parent_low);
    if(s) setFilterRes(1.0/s);            // envelope will cross 0.5 where parent–low does
  } else if(act==="trace"){
    MODE = "ramped"; extra = {trace_profile:true};
  } else {
    MODE = act;                            // ramped | amplitude | cutoff
  }
  onModeChange(MODE);
  runStep("filter", extra);
}
function onModeChange(mode){
  MODE = mode;
  const el = $("#step-filter");
  applyModeVisibility(el);
  configureSteps();                                   // show/hide the amplitude panel for the mode
  const res = $("[data-result]", el); if(res){ disposeViewersIn(res); res.innerHTML = ""; }
  ["amplitude","compare"].forEach(n => { if(!$("#step-"+n).hidden) renderStepShell(n, false); });
}

function renderStepShell(name, active){
  const el = $("#step-"+name);
  disposeViewersIn(el);
  el.classList.toggle("active", active);
  const inner = name === 'register' ? `<div class="filterctl">
    <label for="resampleGrid">Resample grid (voxels on the longest axis)</label>
    <input type="number" id="resampleGrid" min="0" max="256" step="1" value="${HAS_REF||HIGH_DIM<=128?0:128}" style="max-width:140px">
    <div class="hint">${HAS_REF?'0 (default) keeps the reference grid; the parent is resampled onto it.':'Optional: go directly to Filter to use the input grid, or run Step 1 to resample the high map first (default 128 voxels for larger maps, which keeps several seeds within browser memory); 0 keeps the high-map grid.'} A smaller number gives a coarser, faster working grid; maps are low-passed before downsampling to avoid aliasing. Later steps and downloads use this grid.</div>
    </div><div data-result></div><div data-viewers></div>` : (name === "filter") ? filterControlsHTML() : "";
  el.innerHTML = `
    <div class="head" data-head>
      <div style="display:flex;align-items:center;gap:10px"><span class="caret">▾</span><h2>${name==='register'&&!HAS_REF?'Step 1 · resample (optional)':el.dataset.title}</h2></div>
      <div style="display:flex;align-items:center;gap:12px">
        <span class="badge" data-badge>idle</span>
        <button data-run ${active?"":"disabled"}>Run</button>
      </div>
    </div>
    <div data-body>${inner}</div>`;
  $("[data-head]", el).addEventListener("click", (e) => {
    if(e.target.closest("[data-run]")) return;            // the Run button never toggles the panel
    setCollapsed(el, !el.classList.contains("collapsed"));
  });
  $("[data-run]", el).addEventListener("click", (ev) => { ev.stopPropagation(); runStep(name); });
  if(name === "filter"){
    FILTER_VIEWS = null;   // panel (and its [data-viewers]) is fresh; next run builds the pair
    // the mode radios / slider / action buttons drive the run, so hide the generic Run
    const run = $("[data-run]", el); if(run) run.textContent = "Apply filter";
    $$('input[name="fmode"]', el).forEach(r =>
      r.addEventListener("change", () => { if(r.checked) handleFilterAct(r.value); }));   // pick a mode → run
    $$(".fscacts button", el).forEach(b =>
      b.addEventListener("click", () => handleFilterAct(b.dataset.act)));
    const num = $("#fTargetRes", el), rng = $("#fResRange", el);
    if(rng && num){
      rng.addEventListener("input", () => { num.value = rng.value; });                    // live label while dragging
      rng.addEventListener("change", () => { runStep("filter", {}); });                    // re-filter on release
      num.addEventListener("input", () => {
        const v = parseFloat(num.value);
        if(isFinite(v)) rng.value = Math.min(parseFloat(rng.max), Math.max(parseFloat(rng.min), v));
      });
      num.addEventListener("change", () => { runStep("filter", {}); });                    // re-filter on typed value
    }
    ['#fParentRes','#fKappa','#fSeeds'].forEach(selector => {
      $(selector, el).addEventListener('change', () => { if(FILTER_ACTIVE) runStep('filter', {}); });
    });
    $$('.filterctl input[type="number"]', el).forEach(input => {
      input.addEventListener('keydown', event => { if(event.key==='Enter') { event.preventDefault(); input.blur(); } });
    });
    setFilterGrid(INPUT_VOXEL);
    applyModeVisibility(el);
    setFilterEnabled(active);
  }
  enhance(el);
  setCollapsed(el, !active);                              // active step expanded, the rest collapsed
}

// expand/collapse a collapsible section; NGL canvases are zero-sized while
// hidden, so restore their size when the section is re-opened
function setCollapsed(el, collapsed){
  el.classList.toggle("collapsed", collapsed);
  const head=$(".head",el); if(head) head.setAttribute("aria-expanded",String(!collapsed));
  if(!collapsed){
    requestAnimationFrame(() => el.querySelectorAll(".viewer").forEach(h => {
      const v = VIEWERS[h.id]; if(v && v.resize) v.resize();
    }));
  }
}
function filterPayload(extra){
  const el = $("#step-filter");
  const p = {mode: MODE, target_res: parseFloat($("#fTargetRes", el).value || "10")};
  if(MODE !== "cutoff"){
    p.parent_res = parseFloat($("#fParentRes", el).value || "0");
    p.kappa = parseFloat($("#fKappa", el).value || "100");
  }
  if(MODE === "ramped") p.seeds = Number($("#fSeeds", el).value || "4");
  return Object.assign(p, extra || {});
}
async function runStep(name, extra){
  if(BUSY) return;
  busy(true);
  if(name==="register") PRODUCED={aligned:false,filtered:false,matched:false};
  if(name==="filter") {PRODUCED.filtered=false;PRODUCED.matched=false;LAST_FSC=null;}
  if(name==="amplitude") PRODUCED.matched=false;
  const el = $("#step-"+name);
  const badge = $("[data-badge]", el); const btn = $("[data-run]", el);
  const out = $("[data-result]", el) || $("[data-body]", el);   // filter keeps its controls above the result
  const idx = STEP_ORDER.indexOf(name);
  const fileStages=['register','filter','amplitude','compare'], fileIndex=fileStages.indexOf(name);
  compute.clear(file => file.startsWith('results/') || file === 'phaselp_results.zip' || file === 'qc_summary.png' ||
    fileStages.slice(fileIndex).some((stage, offset) => file.startsWith('step'+(fileIndex+offset+1)) || file === 'qc_'+stage+'.png'));
  // re-running a step invalidates everything after it
  for(let k = idx + 1; k < STEP_ORDER.length; k++) renderStepShell(STEP_ORDER[k], false);
  setCollapsed(el, false);                       // make sure the running step is expanded
  badge.className = "badge run"; badge.textContent = "running"; btn.disabled = true;
  if(name === "filter") setFilterEnabled(false);
  disposeViewersIn(out);
  out.innerHTML = '<p class="muted"><span class="spinner"></span> computing…</p>';
  try{
    const payload = name === 'register' ? {resample_dimension:Number($('#resampleGrid').value), ...(extra||{})}
      : (name === "filter") ? filterPayload(extra) : (extra || {});
    const j = await compute.call(name, payload);
    window.lastResult=j;
    if(name==='register' && j.working_grid){
      setFilterGrid(j.working_grid.voxel);
    }
    renderStepResult(name, el, j);
    badge.className = "badge done"; badge.textContent = "done";
    // record what this step produced (filter + amplitude are optional)
    if(name === "register") PRODUCED = {aligned:true, filtered:false, matched:false};
    else if(name === "filter") PRODUCED = {aligned:PRODUCED.aligned, filtered:true, matched:false};
    else if(name === "amplitude") PRODUCED = {aligned:PRODUCED.aligned, filtered:PRODUCED.filtered, matched:true};
    updateEnable();
    if(idx+1 < STEP_ORDER.length) setCollapsed($("#step-"+STEP_ORDER[idx+1]), false);  // guide forward (optional)
  }catch(e){
    badge.className="badge err"; badge.textContent="error";
    out.innerHTML = `<div class="err">${escapeHTML(errorMessage(e))}</div>`;
    const viewers=$('[data-viewers]',el);
    if(viewers){ disposeViewersIn(viewers); viewers.innerHTML=''; }
    if(name==='filter') FILTER_VIEWS=null;
  }finally{ busy(false); updateEnable(); enhance(el); }
}
function metricsTable(m){
  return '<table class="metrics">' + Object.entries(m).map(([k,v]) =>
    `<tr data-metric="${escapeHTML(k)}"><td>${escapeHTML(k.replace(/_/g," "))}</td><td>${escapeHTML(Array.isArray(v)?v.join(", "):v)}</td></tr>`).join("") + "</table>";
}
// wrap the plots + metrics + downloads in a collapsible block so the 3-D
// viewer can stand alone; the viewer itself lives OUTSIDE this block
function analyticsBlock(inner){
  return `<div class="analytics collapsed" data-analytics>
      <div class="anhead" data-anhead><span class="ancaret">▾</span> plots &amp; metrics</div>
      <div class="anbody">${inner}</div>
    </div>`;
}
function wireAnalytics(scope){
  const a = $("[data-analytics]", scope); if(!a) return;
  $("[data-anhead]", a).addEventListener("click", () => a.classList.toggle("collapsed"));
}
function renderStepResult(name, el, j){
  const body = $("[data-result]", el) || $("[data-body]", el);   // filter keeps its controls above
  disposeViewersIn(body);
  const dls = (j.downloads || (j.download? [j.download] : []))
      .map(d => `<a href="${withSid(d.url)}" download="${escapeHTML(compute.name(d.url).split("/").pop())}">${escapeHTML(d.label)}</a>`).join("");
  if(name === "filter" && j.fsc_chart){
    LAST_FSC = j.fsc_chart;                       // remembered for "Construct to match"
    body.innerHTML = analyticsBlock(`
      <div class="fscrow">
        <img class="qcimg2" src="${withSid(j.plot)}" alt="filter QC">
        <div class="fscchart" data-chart></div>
      </div>
      ${metricsTable(j.metrics||{})}
      <div class="dl">${dls}</div>`);
    renderFscChart($("[data-chart]", body), j.fsc_chart);
    wireAnalytics(body);
  } else {
    body.innerHTML = analyticsBlock(`
      <img class="qcimg" src="${withSid(j.plot)}" alt="QC plot">${j.summary_plot?`<img class="qcimg" src="${withSid(j.summary_plot)}" alt="Pipeline summary">`:""}
      ${metricsTable(j.metrics||{})}
      <div class="dl">${dls}</div>`) + `<div data-pair></div>`;
    wireAnalytics(body);
  }
  const plotNames=[compute.name(j.plot), ...(j.summary_plot?[compute.name(j.summary_plot)]:[])];
  $$('.qcimg,.qcimg2',body).forEach((img,i)=>attachPlot(img,j.plot_data?.[plotNames[i]]));
  enhance(el);
  const items = [];
  if(j.viewer) items.push({url:j.viewer, label:j.viewer_label || "result"});
  if(j.reference_viewer) items.push({url:j.reference_viewer, label:j.reference_label || "reference"});
  if(name === "filter"){
    // persistent pair: reload the map in place so the viewport is kept across re-runs
    if(items.length) renderFilterViewers($("[data-viewers]", el), items);
  } else if(items.length){
    mountPair($("[data-pair]", el), items);
  }
}

// Interactive FSC chart for the filter step. Two shapes:
//   with a reference    : {has_reference:true, parent_low, filtered_low, envelope, target_res, mode}
//       primary = FSC(parent, reference); secondary = FSC(filtered, reference);
//   without a reference : {has_reference:false, self_filtered, envelope, target_res, mode}
//       primary = FSC(parent, filtered).
//   envelope = the coherence c(s) the filter applied (eq. 5 or 7, the traced curve,
//   or the kept band of a hard cutoff), computed in Python; mode = model | traced | cutoff.
// The action buttons live in the toolbar above; this chart handles hover + click-to-retarget.
function renderFscChart(container, data){
  const NS = "http://www.w3.org/2000/svg";
  const W = 460, H = 300, ml = 46, mr = 14, mt = 14, mb = 38;
  const hasRef = data.has_reference !== false;
  // primary curve: parent–low (with a reference) or high-vs-filtered (without)
  const primary = hasRef ? (data.parent_low || {freq:[],fsc:[]}) : (data.self_filtered || {freq:[],fsc:[]});
  const secondary = hasRef ? (data.filtered_low || {freq:[],fsc:[]}) : {freq:[],fsc:[]};
  const pf = primary.freq || [], pv = primary.fsc || [];
  const sf = secondary.freq || [], sv = secondary.fsc || [];
  const ef = (data.envelope && data.envelope.freq) || [], ev = (data.envelope && data.envelope.value) || [];
  const empirical = data.mode === "traced";
  const isCut = data.mode === "cutoff";
  let targetRes = data.target_res || 10;
  const allf = pf.concat(sf);
  const xmax = allf.length ? Math.max(...allf) : 0.5;
  const ymin = -0.2, ymax = 1.05;
  const C = {grid:"#9aa4b2", tick:"#8892a0", parent:"#2a86c8", surr:"#9aa9bb", env:"#d55e00", cross:"#e0663a", match:"#3aa657"};
  const xpix = s => ml + (s / xmax) * (W - ml - mr);
  const ypix = v => mt + (1 - (v - ymin) / (ymax - ymin)) * (H - mt - mb);
  const mk = (t, a) => { const e = document.createElementNS(NS, t); for(const k in a) e.setAttribute(k, a[k]); return e; };
  const svg = mk("svg", {viewBox:`0 0 ${W} ${H}`, class:"fscsvg"});

  svg.appendChild(mk("rect", {x:ml, y:mt, width:W-ml-mr, height:H-mt-mb, fill:"none", stroke:C.grid, "stroke-width":0.8}));
  [0, 0.143, 0.5, 1].forEach(v => {
    const y = ypix(v), dash = (v===0.5||v===0.143);
    svg.appendChild(mk("line", {x1:ml, y1:y, x2:W-mr, y2:y, stroke:C.grid,
      "stroke-dasharray": dash?"3 3":"", "stroke-width": dash?1:0.5}));
    const t = mk("text", {x:ml-6, y:y+3, "text-anchor":"end", "font-size":10, fill:C.tick}); t.textContent = v; svg.appendChild(t);
  });
  const nT = 5;
  for(let i=0;i<=nT;i++){ const s = xmax*i/nT, x = xpix(s);
    svg.appendChild(mk("line", {x1:x, y1:H-mb, x2:x, y2:H-mb+4, stroke:C.tick}));
    const t = mk("text", {x:x, y:H-mb+16, "text-anchor":"middle", "font-size":10, fill:C.tick}); t.textContent = s.toFixed(3); svg.appendChild(t);
  }
  const xlab = mk("text", {x:(ml+W-mr)/2, y:H-4, "text-anchor":"middle", "font-size":10, fill:C.tick}); xlab.textContent = "spatial frequency (1/Å)"; svg.appendChild(xlab);
  const yc = (mt + H - mb) / 2;
  const ylab = mk("text", {x:13, y:yc, "text-anchor":"middle", "font-size":11, fill:C.tick, transform:`rotate(-90 13 ${yc})`}); ylab.textContent = "FSC"; svg.appendChild(ylab);

  const line = (f, v, stroke, width, dash) => {
    if(!f.length) return;
    const pts = f.map((s,i)=>`${xpix(s)},${ypix(v[i])}`).join(" ");
    svg.appendChild(mk("polyline", Object.assign({fill:"none", points:pts, stroke, "stroke-width":width}, dash?{"stroke-dasharray":dash}:{})));
  };
  // secondary first (drawn under), then primary on top
  if(hasRef) line(sf, sv, C.surr, 1.3);
  line(pf, pv, C.parent, 2.0);

  const envPath = mk("polyline", {fill:"none", stroke:C.env, "stroke-width":1.8, "stroke-dasharray":"5 3"}); svg.appendChild(envPath);
  const targetLine = mk("line", {y1:mt, y2:H-mb, stroke:C.tick, "stroke-dasharray":"4 3"}); svg.appendChild(targetLine);
  // c(s) as applied by the filter (sent by Python), linearly interpolated for hover readouts
  const envAt = s => {
    if(!ef.length) return NaN;
    if(s <= ef[0]) return ev[0];
    if(s >= ef[ef.length-1]) return ev[ev.length-1];
    let i = 1; while(i < ef.length && ef[i] < s) i++;
    const t = (s - ef[i-1]) / (ef[i] - ef[i-1]); return ev[i-1] + t*(ev[i]-ev[i-1]);
  };
  function drawEnv(){
    const pts = [];
    if(isCut){
      // hard cutoff: the kept band s <= 1/R
      const xc = xpix(1/targetRes);
      pts.push(`${xpix(0)},${ypix(1)}`, `${xc},${ypix(1)}`, `${xc},${ypix(0)}`, `${xpix(xmax)},${ypix(0)}`);
    } else {
      for(let i=0;i<ef.length;i++){ if(ef[i] <= xmax && isFinite(ev[i])) pts.push(`${xpix(ef[i])},${ypix(ev[i])}`); }
    }
    const x = xpix(1/targetRes); targetLine.setAttribute("x1", x); targetLine.setAttribute("x2", x);
    targetLine.setAttribute("opacity", empirical ? 0 : 1);
    envPath.setAttribute("points", pts.join(" "));
  }
  drawEnv();

  // marker at the parent-low 0.5 crossing (what "Construct to match" would pick)
  const crossing = (f, v, level) => {
    for(let i=1;i<f.length;i++){ const a=v[i-1], b=v[i];
      if(isFinite(a)&&isFinite(b)&&a>=level&&b<level){ const t=(level-a)/(b-a); return f[i-1]+t*(f[i]-f[i-1]); } }
    return null;
  };
  const sHalf = crossing(pf, pv, 0.5);
  if(sHalf){
    const x = xpix(sHalf);
    svg.appendChild(mk("line", {x1:x, y1:mt, x2:x, y2:H-mb, stroke:C.match, "stroke-width":1, "stroke-dasharray":"2 3", opacity:0.9}));
    const t = mk("text", {x:x, y:mt+11, "text-anchor":"middle", "font-size":9, fill:C.match}); t.textContent = `${(1/sHalf).toFixed(1)} Å`; svg.appendChild(t);
  }

  const legend = [[C.parent, hasRef ? "— FSC(parent, reference)" : "— FSC(parent, filtered)"]];
  if(hasRef) legend.push([C.surr, "— FSC(filtered, reference)"]);
  legend.push([C.env, empirical ? "-- c(s) = traced parent–reference FSC" : (isCut ? "-- hard cutoff" : "-- filter c(s), eq. 7")]);
  // legend in the upper right, where every curve has already fallen towards zero
  legend.forEach((L, i) => {
    const t = mk("text", {x:W-mr-8, y:mt+14+i*13, "font-size":10, fill:L[0], "text-anchor":"end"}); t.textContent = L[1]; svg.appendChild(t);
  });

  const vline = mk("line", {y1:mt, y2:H-mb, stroke:C.cross, "stroke-width":0.8, opacity:0}); svg.appendChild(vline);
  const hline = mk("line", {x1:ml, x2:W-mr, stroke:C.cross, "stroke-width":0.8, opacity:0}); svg.appendChild(hline);
  const rbg = mk("rect", {fill:"rgba(0,0,0,0.65)", rx:4, width:132, height:34, opacity:0}); svg.appendChild(rbg);
  const r1 = mk("text", {"font-size":11, fill:"#fff", opacity:0}); svg.appendChild(r1);
  const r2 = mk("text", {"font-size":11, fill:"#fff", opacity:0}); svg.appendChild(r2);
  const ovl = mk("rect", {x:ml, y:mt, width:W-ml-mr, height:H-mt-mb, fill:"transparent", class:"ovl"}); svg.appendChild(ovl);

  const interp = s => {   // read the primary (parent-low) curve
    if(!pf.length) return NaN;
    if(s <= pf[0]) return pv[0];
    if(s >= pf[pf.length-1]) return pv[pf.length-1];
    let i = 1; while(i < pf.length && pf[i] < s) i++;
    const t = (s - pf[i-1]) / (pf[i] - pf[i-1]); return pv[i-1] + t*(pv[i]-pv[i-1]);
  };
  const sFromEvent = ev => { const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (W / r.width);
    return Math.max(0, Math.min(xmax, (px - ml)/(W-ml-mr)*xmax)); };
  ovl.addEventListener("mousemove", ev => {
    const s = sFromEvent(ev), v = interp(s), x = xpix(s), y = ypix(v);
    vline.setAttribute("x1", x); vline.setAttribute("x2", x); vline.setAttribute("opacity", 0.7);
    hline.setAttribute("y1", y); hline.setAttribute("y2", y); hline.setAttribute("opacity", 0.7);
    r1.textContent = `res = ${s>0 ? (1/s).toFixed(1) : "∞"} Å`;
    r2.textContent = `FSC = ${isFinite(v) ? v.toFixed(3) : "–"}`;
    const interpolate=(f,a)=>{if(!f.length)return NaN;let i=1;while(i<f.length&&f[i]<s)i++;if(i>=f.length)return a[a.length-1];if(s<=f[0])return a[0];const t=(s-f[i-1])/(f[i]-f[i-1]);return a[i-1]+t*(a[i]-a[i-1]);};
    const fmt=v=>Number.isFinite(v)?v.toFixed(4):'—';
    const readings=[`Frequency ${s.toFixed(4)} Å⁻¹ · resolution ${s>0?(1/s).toFixed(2):'∞'} Å`,`${hasRef?'Parent vs reference':'Parent vs filtered'} FSC: ${fmt(v)}`];
    if(hasRef)readings.push(`Filtered vs reference FSC: ${fmt(interpolate(sf,sv))}`);
    readings.push(`Filter c(s): ${fmt(isCut?(s<=1/targetRes?1:0):envAt(s))}`,'FSC guides: 0.5 and 0.143 · click to set filter target');
    inspect.show(ev,'Fourier shell correlation',readings);
    const bx = Math.min(x+8, W-mr-134), by = Math.max(mt+2, y-40);
    rbg.setAttribute("x", bx); rbg.setAttribute("y", by); rbg.setAttribute("opacity", 1);
    r1.setAttribute("x", bx+8); r1.setAttribute("y", by+14); r1.setAttribute("opacity", 1);
    r2.setAttribute("x", bx+8); r2.setAttribute("y", by+28); r2.setAttribute("opacity", 1);
  });
  ovl.addEventListener("mouseleave", () => {inspect.hide();[vline,hline,rbg,r1,r2].forEach(e => e.setAttribute("opacity", 0));});
  ovl.addEventListener("click", ev => {
    const s = sFromEvent(ev); if(s <= 0 || BUSY) return;
    setFilterRes(1.0 / s);            // move the "Filter to" control to the clicked resolution
    runStep("filter", {});            // and re-filter there in the current mode
  });

  container.innerHTML = ""; container.appendChild(svg);
  const cap = document.createElement("div"); cap.className = "hint";
  cap.textContent = hasRef
    ? "Blue = parent vs reference; grey = filtered map vs reference, expected to be c(s) × the blue curve (Section S2). Hover to read values; click a frequency to re-filter there. Green marks where the parent–reference FSC crosses 0.5 (what “Construct to match” picks)."
    : "Blue = parent vs its filtered map, expected to follow c(s) (eq. 4). Hover to read values; click a frequency to re-filter there.";
  container.appendChild(cap);
}

// Startup uses only relative, bundled real maps, independent of EMDB CORS.
async function startEngine(){
  try {
    const result=await compute.call('proof');
    if(!result.ok)throw new Error('Compute engine FSC self-check failed.');
    $('#engine').classList.add('ready');
    $('#engineStatus').textContent='Ready · computation and files stay in your browser';
    const pageURL=new URL(location.href);
    if(pageURL.searchParams.has('reset')){
      // Firefox restores form/file selections on navigation. Apply reset after
      // startup, when that restoration has finished, before loading any input.
      $$('#inputPanel input').forEach(input=>{
        if(input.type==='checkbox') input.checked=input.defaultChecked;
        else input.value=input.type==='file'?'':input.defaultValue;
      });
      $$('#inputPanel select').forEach(select=>{
        select.value=([...select.options].find(option=>option.defaultSelected)||select.options[0]).value;
        select.dispatchEvent(new Event('change'));
      });
      updRefVisibility();
      pageURL.searchParams.delete('reset'); history.replaceState(null,'',pageURL);
    }
    $('#loadBtn').disabled=false;
    if(!new URLSearchParams(location.search).has('no-demo')) $('#loadBtn').click();
  }catch(e){$('#inputErr').hidden=false;$('#inputErr').textContent='Could not load the compute engine. Check your connection and reload. '+errorMessage(e);}
}
$('#engineCheck').addEventListener('click',async()=>{
  const button=$('#engineCheck');button.disabled=true;
  try{const r=await compute.call('proof');$('#engineStatus').textContent=`FSC check ${r.ok?'passed':'failed'} · NumPy ${r.numpy} · SciPy ${r.scipy} · mrcfile ${r.mrcfile}`;}
  catch(e){$('#engineStatus').textContent=errorMessage(e);}finally{button.disabled=false;}
});
enhance();
startEngine();
