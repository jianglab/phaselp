const tooltip = document.createElement('div');
tooltip.className = 'inspect-tooltip'; tooltip.id = 'inspect-tooltip';
tooltip.setAttribute('role','tooltip'); tooltip.hidden = true;
document.body.appendChild(tooltip);
export const inspect = {
  show(event, title, lines=[]) {
    tooltip.replaceChildren();
    const heading=document.createElement('strong'); heading.textContent=title; tooltip.append(heading);
    for(const line of lines) { const row=document.createElement('div'); row.textContent=line; tooltip.append(row); }
    tooltip.hidden=false;
    const x=event.clientX ?? event.target.getBoundingClientRect().left;
    const y=event.clientY ?? event.target.getBoundingClientRect().bottom;
    tooltip.style.left=Math.max(8,Math.min(x+14,innerWidth-tooltip.offsetWidth-10))+'px';
    tooltip.style.top=Math.max(8,Math.min(y+14,innerHeight-tooltip.offsetHeight-10))+'px';
  },
  hide(){tooltip.hidden=true;}
};
document.addEventListener('keydown', e=>{if(e.key==='Escape') inspect.hide();});
window.addEventListener('scroll',()=>inspect.hide(),true);
const descriptions = {
  registered:'Whether similarity registration was run. Skipping still resamples the parent onto the reference grid.',
  ncc_before:'Band-limited normalized cross-correlation on the optimization grid before registration.',
  ncc_after:'Band-limited normalized cross-correlation after registration.',
  scale:'Fitted isotropic scale factor. A value of 1 preserves the parent scale.',
  rotation_deg:'Angle of the fitted rotation, in degrees.',
  translation_zyx_vox:'Fitted translation in working-grid voxels, ordered z, y, x.',
  working_grid_zyx:'Shape of the working grid after Step 1, ordered z, y, x. Later steps and downloads use this grid.',
  working_voxel_A:'Voxel size of the working grid in Å.',
  mode:'Ramped phase: amplitude response T = c plus wrapped-normal phase offsets (eq. 2, 5–7). Amplitude only: the same T without offsets. Hard cutoff: keep s ≤ 1/R.',
  filter_res_A:'Target resolution R in Å. The filter coherence c(s) is 0.5 at s = 1/R.',
  n_seeds:'Number of independent filtered maps (one per seed). Use each map on its own; average statistics, not maps.',
  kappa:'SSNR scale κ of eq. 5 (100 in the manuscript).',
  parent_res_A:'Parent resolution used by eq. 7. Zero treats the parent as ideal (eq. 5).',
  fsc_parent_filtered_at_target:'FSC between the parent and its filtered map at the target shell, mean over seeds.',
  expected_c_at_target:'The coherence c(s) of the filter at the target shell: the expected FSC with the parent (eq. 4).',
  fsc_parent_reference_at_target:'FSC between the registered parent and the reference at the target shell.',
  expected_parent_reference_at_target:'Eq. 8: a_H·a_T from the nominal resolutions (a_T alone for an ideal parent).',
  fsc_filtered_reference_at_target:'FSC between the filtered maps and the reference at the target shell, mean over seeds.',
  expected_filtered_reference_at_target:'Section S2: c(s) times the unfiltered parent–reference FSC.',
  residual_variance_in_mask:'Variance of (filtered − reference) inside a soft mask of the reference (mask > 0.5), mean over seeds. Divide by the value from a same-state control to get the ratio of Section 3.2.',
  local_rms_p95:'95th percentile, inside the mask, of the local rms of one difference map (Gaussian σ = target resolution).',
  max_local_rms:'Largest local rms of one difference map inside the mask.',
  compared:'Which maps were compared with the reference.',
  n_maps:'Number of maps that were amplitude matched.',
  max_gain:'Maximum per-shell amplitude gain; limits amplification of weak Fourier shells.',
};
const tips = {
  '.iso':'Isosurface threshold in standard deviations (σ) above the viewer-map mean. Lower values reveal weaker density.',
  '.depth':'Position of the screen-parallel section through the volume. 50% is the center; rotate the 3-D view to change the plane.',
  '.synced':'Couples camera orientation, isosurface threshold and slice depth between these two maps.',
  '#fResRange, #fTargetRes':'Target resolution in Å. Increasing it removes finer detail. Release the slider or edit the number to run.',
  '#fParentRes':'Parent resolution in Å for eq. 7, which corrects for the parent’s own loss of correlation. Zero treats the parent as ideal.',
  '#fKappa':'SSNR scale κ of eq. 5. Must be greater than 1/3; the manuscript uses 100.',
  '#fSeeds':'Independent filtered maps with seeds 0…N−1. More seeds give tighter averages of the statistics and use more memory.',
  '[value="ramped"]':'Ramped phase randomization: amplitude response T = c(s) and wrapped-normal phase offsets with coherence c(s).',
  '[value="amplitude"]':'Amplitude-only control: the same amplitude response, every phase of the parent kept.',
  '[value="cutoff"]':'Hard cutoff: keep Fourier coefficients with s ≤ 1/R and remove the rest (Figure 1).',
  '[data-act="construct"]':'Requires a reference map. Set the target to where the parent–reference FSC crosses 0.5, then run ramped phase filtering.',
  '[data-act="trace"]':'Requires a reference map. Use the measured parent–reference FSC as the coherence profile c(s), then run ramped phase filtering.',
  '#orient':'Global: start from the best of 24 principal-axis orientations or the identity (Section 5.1). Local: start from the identity only.',
  '#maxDim':'Largest dimension of the optimizer’s downsampled search grid (96 in the manuscript). The registered map stays on the working grid.',
  '#resampleGrid':'Working grid for Step 1 and every later step. 0 keeps the reference grid (or the high-map grid without a reference). Coarser grids are low-passed before downsampling. Every rerun starts from the original inputs.',
  '#alignRes':'Band limit of the registration NCC in Å. The manuscript uses the target resolution.',
  '#standardize':'Subtract each input mean and divide by its standard deviation before processing (Section 5.3).',
  '#skipReg':'Resample the high map to the reference grid without fitting scale, orientation or translation.',
  '#emdbCap':'Resample downloaded EMDB inputs (anti-aliased) to this maximum dimension. Zero preserves the original grid, subject to browser memory limits. Uploaded maps keep their original grid.',
};
function help(el,text) {
  if(el.dataset.helpBound) return;
  el.dataset.helpBound='1'; el.dataset.help=text;
  if(!el.matches('input,button,a,select')) el.tabIndex=0;
  el.setAttribute('aria-description',text);
  el.addEventListener('pointerenter',e=>inspect.show(e,text));
  el.addEventListener('pointerleave',inspect.hide);
  el.addEventListener('focus',e=>inspect.show(e,text));
  el.addEventListener('blur',inspect.hide);
}
export function enhance(scope=document) {
  for(const [sel,text] of Object.entries(tips)) scope.querySelectorAll(sel).forEach(el=>help(el,text));
  scope.querySelectorAll('[data-metric]').forEach(el=>{
    const k=el.dataset.metric;
    help(el,descriptions[k] || el.textContent);
  });
  scope.querySelectorAll('.dl a').forEach(el=>help(el,'Download '+el.textContent+'. Generated locally in this browser.'));
  scope.querySelectorAll('.head, .anhead').forEach(el=>{
    if(el.dataset.keyboard) return; el.dataset.keyboard='1'; el.tabIndex=0;
    el.setAttribute('role','button');
    el.addEventListener('keydown',e=>{if(e.target===el && (e.key==='Enter'||e.key===' ')){e.preventDefault();el.click();}});
  });
  scope.querySelectorAll('.slice2d').forEach(canvas=>{
    if(canvas.dataset.inspect) return; canvas.dataset.inspect='1';
    canvas.addEventListener('pointermove',e=>{
      const s=canvas.sliceInfo;if(!s) return;
      const rect=canvas.getBoundingClientRect();
      const side=Math.min(rect.width,rect.height), left=rect.left+(rect.width-side)/2, top=rect.top+(rect.height-side)/2;
      const x=Math.floor((e.clientX-left)/side*s.size), y=Math.floor((e.clientY-top)/side*s.size);
      if(x<0||y<0||x>=s.size||y>=s.size){inspect.hide();return;}
      const density=s.values[y*s.size+x], xyz=s.coords(x,y), v=s.volume;
      inspect.show(e,'Density section',[
        `Density ${density.toPrecision(5)} · ${v.sigma?((density-v.mean)/v.sigma).toFixed(2):'0'} σ`,
        `Grid x, y, z: ${xyz.map(n=>n.toFixed(1)).join(', ')}`,
        `Depth ${s.depth}% · offset ${(s.offset*v.voxel).toFixed(1)} Å`,
        `Viewer sampling ${v.voxel.toFixed(3)} Å/voxel`]);
    });
    canvas.addEventListener('pointerleave',inspect.hide);
  });
}
const number=x=>Number.isFinite(x)?(Math.abs(x)<.001&&x!==0?x.toExponential(3):x.toPrecision(5)):'—';
export function attachPlot(img, axes) {
  img.tabIndex=0; img.setAttribute('aria-description','Hover over a plot panel to inspect its plotted coordinates and values.');
  img.addEventListener('pointermove',event=>{
    const r=img.getBoundingClientRect(), u=(event.clientX-r.left)/r.width, v=1-(event.clientY-r.top)/r.height;
    const ax=axes?.find(a=>u>=a.bounds[0]&&u<=a.bounds[0]+a.bounds[2]&&v>=a.bounds[1]&&v<=a.bounds[1]+a.bounds[3]);
    if(!ax){inspect.hide();return;}
    const coordinate=(t,limits,scale)=>scale==='log'?Math.exp(Math.log(limits[0])+t*(Math.log(limits[1])-Math.log(limits[0]))):limits[0]+t*(limits[1]-limits[0]);
    const x=coordinate((u-ax.bounds[0])/ax.bounds[2],ax.xlim,ax.xscale);
    const y=coordinate((v-ax.bounds[1])/ax.bounds[3],ax.ylim,ax.yscale);
    const lines=[`${ax.xlabel||'x'}: ${number(x)}`,`${ax.ylabel||'y'}: ${number(y)}`];
    if(/frequency/i.test(ax.xlabel)&&x>0) lines.push(`Resolution ${(1/x).toFixed(2)} Å`);
    for(const line of ax.lines) {
      if(line.label.startsWith('_')) continue;
      let nearest=0;
      for(let i=1;i<line.x.length;i++) if(Math.abs(line.x[i]-x)<Math.abs(line.x[nearest]-x)) nearest=i;
      lines.push(`${line.label}: ${number(line.y[nearest])}`);
    }
    for(const im of ax.images) {
      const [left,right,bottom,top]=im.extent, [h,w]=im.shape;
      const col=Math.floor((x-left)/(right-left)*w);
      const row=Math.floor((im.origin==='upper'?(top-y)/(top-bottom):(y-bottom)/(top-bottom))*h);
      if(row>=0&&row<h&&col>=0&&col<w) lines.push(`Pixel (${col}, ${row}): ${number(im.values[row*w+col])}`);
    }
    inspect.show(event,ax.title||'QC plot',lines.slice(0,12));
  });
  img.addEventListener('pointerleave',inspect.hide);
}
