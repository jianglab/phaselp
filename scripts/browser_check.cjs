// End-to-end check of the phaselp browser app (apps/) in Chromium with Playwright.
//
//   python -m http.server 8765 --bind 127.0.0.1          # from the repository root
//   SITE_URL=http://127.0.0.1:8765/apps/ OUT_DIR=/tmp/phaselp-check node scripts/browser_check.cjs
//
// Runs the bundled demo through all four steps, the other filter modes and the
// no-reference mode, saves screenshots and the results ZIP, and fails on any page
// error. Optional CDN_DIR serves the Pyodide 0.29.3 distribution (CDN_DIR/pyodide)
// and NGL 2.3.1 (CDN_DIR/ngl/package/dist) locally instead of from jsDelivr.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const CDN = process.env.CDN_DIR, OUT = process.env.OUT_DIR || '.', URL_ = process.env.SITE_URL || 'http://127.0.0.1:8765/apps/';
const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.zip': 'application/zip', '.whl': 'application/zip', '.tar': 'application/x-tar' };
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  let served = 0, missing = [];
  if (CDN) await context.route(/^https:\/\/cdn\.jsdelivr\.net\/(pyodide\/v0\.29\.3\/full|npm\/ngl@2\.3\.1\/dist)\/.*/, async route => {
    const u = new URL(route.request().url());
    const file = u.pathname.startsWith('/pyodide/') ? path.join(CDN, 'pyodide', path.basename(u.pathname)) : path.join(CDN, 'ngl/package/dist', path.basename(u.pathname));
    if (!fs.existsSync(file)) { missing.push(u.href); return route.fulfill({ status: 404, body: 'missing' }); }
    served++;
    return route.fulfill({ status: 200, body: fs.readFileSync(file), headers: { 'content-type': types[path.extname(file)] || 'application/octet-stream', 'access-control-allow-origin': '*' } });
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const t0 = Date.now();
  await page.goto(URL_);
  await page.waitForSelector('#engine.ready', { timeout: 300000 });
  log('engine ready', ((Date.now() - t0) / 1000).toFixed(0) + 's', 'cdn files served', served);
  await page.waitForFunction(() => /loaded/.test(document.querySelector('#loadStatus').textContent), null, { timeout: 300000 });
  log('demo loaded', await page.textContent('#highMeta'), '|', await page.textContent('#lowMeta'));
  const metrics = async step => page.$$eval(`#step-${step} table.metrics tr`, rows => Object.fromEntries(rows.map(r => [r.dataset.metric, r.querySelector('td:last-child').textContent])));
  const run = async (step, timeout = 600000) => {
    const t = Date.now();
    await page.click(`#step-${step} [data-run]`);
    await page.waitForFunction(s => ['done', 'error'].includes(document.querySelector(`#step-${s} [data-badge]`).textContent), step, { timeout });
    const badge = await page.textContent(`#step-${step} [data-badge]`);
    if (badge === 'error') throw new Error(step + ': ' + await page.textContent(`#step-${step} .err`));
    log(step, 'done in', ((Date.now() - t) / 1000).toFixed(0) + 's', JSON.stringify(await metrics(step)));
  };
  const results = {};
  await run('register'); results.register = await metrics('register');
  // filter to 11.5 A: type in the number box and commit (change event runs the filter)
  const t = Date.now();
  await page.fill('#fTargetRes', '11.5');
  await page.dispatchEvent('#fTargetRes', 'change');
  await page.waitForFunction(() => document.querySelector('#step-filter [data-badge]').textContent === 'done', null, { timeout: 300000 });
  results.filter = await metrics('filter'); log('filter done in', ((Date.now() - t) / 1000).toFixed(0) + 's', JSON.stringify(results.filter));
  await run('amplitude'); results.amplitude = await metrics('amplitude');
  await run('compare'); results.compare = await metrics('compare');
  // open analytics, check images and the chart, downloads
  await page.$$eval('[data-analytics]', els => els.forEach(e => e.classList.remove('collapsed')));
  const imgs = await page.$$eval('img.qcimg, img.qcimg2', els => els.map(e => ({ ok: e.complete && e.naturalWidth > 0, w: e.naturalWidth })));
  const chart = await page.$$eval('#step-filter svg.fscsvg polyline', els => els.map(e => e.getAttribute('points').split(' ').length));
  const legend = await page.$$eval('#step-filter svg.fscsvg text', els => els.map(e => e.textContent).filter(t => /FSC|c\(s\)|cutoff/.test(t)));
  const links = await page.$$eval('#step-compare .dl a', els => els.map(e => e.textContent));
  log('QC images', JSON.stringify(imgs), '| chart polylines', JSON.stringify(chart), '| legend', JSON.stringify(legend));
  log('downloads', JSON.stringify(links));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#step-compare .dl a:last-child')]);
  const zipPath = path.join(OUT, await dl.suggestedFilename()); await dl.saveAs(zipPath);
  log('zip saved', zipPath, fs.statSync(zipPath).size, 'bytes');
  await page.screenshot({ path: path.join(OUT, 'compare.png'), fullPage: true });
  // other modes from the filter step; amplitude-only then hard cutoff
  for (const mode of ['amplitude', 'cutoff']) {
    await page.click(`#step-filter input[value="${mode}"]`);
    await page.waitForFunction(() => document.querySelector('#step-filter [data-badge]').textContent === 'done', null, { timeout: 300000 });
    results['filter_' + mode] = await metrics('filter'); log('filter', mode, JSON.stringify(results['filter_' + mode]));
  }
  await page.screenshot({ path: path.join(OUT, 'filter_cutoff.png'), fullPage: true });
  await page.click('#step-filter input[value="ramped"]');
  await page.waitForFunction(() => document.querySelector('#step-filter [data-badge]').textContent === 'done', null, { timeout: 300000 });
  await page.$$eval('#step-filter [data-analytics]', els => els.forEach(e => e.classList.remove('collapsed')));
  await (await page.$('#step-filter .fscrow')).screenshot({ path: path.join(OUT, 'filter_chart.png') });
  // no-reference mode: filter the high map alone, without Step 1
  await page.check('#noRef');
  await page.click('#loadBtn');
  await page.waitForFunction(() => /loaded/.test(document.querySelector('#loadStatus').textContent), null, { timeout: 300000 });
  await run('register');
  await page.fill('#fTargetRes', '8');
  await page.dispatchEvent('#fTargetRes', 'change');
  await page.waitForFunction(() => ['done', 'error'].includes(document.querySelector('#step-filter [data-badge]').textContent), null, { timeout: 300000 });
  results.no_reference = await metrics('filter'); log('no reference filter', JSON.stringify(results.no_reference));
  if (!results.no_reference.mode) throw new Error('no-reference filtering failed: ' + await page.textContent('#step-filter'));
  fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 1));
  log('page errors', JSON.stringify(errors), 'missing cdn', JSON.stringify(missing));
  await browser.close();
  if (errors.length) { console.error('FAILED: page errors'); process.exit(1); }
  log('PASSED');
})().catch(e => { console.error('FAILED', e); process.exit(1); });
