/* OCBM live designer \u2014 front/back design picker with instant mockup preview.
   Mount: <div id="ocbm-designer" data-garment="crew|tee|hoodie|camo" data-team="NSE" data-form="product-form-..."></div>
   Choices and uploaded files ride with the order as line-item properties. Config: <base>config.json */
(function () {
  'use strict';
  var mount = document.getElementById('ocbm-designer');
  if (!mount) return;
  var BASE = mount.getAttribute('data-base') || 'https://raw.githubusercontent.com/Itfdff9/ocbm-repository/main/customizer/';
  var GARMENT = mount.getAttribute('data-garment');
  var TEAM = mount.getAttribute('data-team');
  var FORM_ID = mount.getAttribute('data-form');
  var MIN_DPI_OK = 150, MIN_DPI_WARN = 100;

  var cfg = null, team = null, gar = null;
  var imgCache = {};
  var state = {
    view: 'front',
    color: null,
    front: { design: null, loc: null, file: null, upImg: null },
    back: { design: null, loc: null, file: null, upImg: null, name: '', number: '' }
  };

  /* ---------- helpers ---------- */
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'style') e.style.cssText = attrs[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  }
  function loadImg(src) {
    if (imgCache[src]) return imgCache[src];
    imgCache[src] = new Promise(function (res, rej) {
      var i = new Image(); i.crossOrigin = 'anonymous';
      i.onload = function () { res(i); }; i.onerror = function () { delete imgCache[src]; rej(new Error('load ' + src)); };
      i.src = src;
    });
    return imgCache[src];
  }
  function fmt(n) { return (Math.round(n * 10) / 10).toString(); }

  /* ---------- persistent form fields (live outside the section so a re-render can't wipe them) ---------- */
  var holder = document.getElementById('ocbm-designer-fields');
  if (!holder) {
    holder = el('div', { id: 'ocbm-designer-fields', style: 'position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden', 'aria-hidden': 'true' });
    document.body.appendChild(holder);
  }
  function field(name, type) {
    var id = 'ocbm-f-' + name.replace(/\W+/g, '-');
    var f = document.getElementById(id);
    if (!f) {
      f = el('input', { id: id, type: type || 'hidden', name: 'properties[' + name + ']', form: FORM_ID, tabindex: '-1' });
      if (type === 'file') f.setAttribute('accept', 'image/png,image/jpeg,image/webp');
      holder.appendChild(f);
    }
    return f;
  }
  function setField(name, value) {
    var f = field(name);
    if (value) { f.value = value; f.disabled = false; } else { f.value = ''; f.disabled = true; }
  }

  /* ---------- current color + base photo ---------- */
  function readColor() {
    var sel = document.querySelector('select[name="options[Color]"]');
    if (sel) return sel.value;
    var r = document.querySelector('input[name^="Color"]:checked, input[type="radio"][name*="Color"]:checked');
    return r ? r.value : null;
  }
  function baseFor(view) {
    var c = state.color;
    if (GARMENT === 'crew' || GARMENT === 'tee') {
      var entry = gar.colors[c] || gar.colors[gar.fallback];
      var hex = (gar.colors[c] && gar.colors[c].hex) || (gar.hexFallback && gar.hexFallback[c]) || '#222222';
      if (view === 'front') return { img: BASE + entry.front, ref: gar.front.ref, ppi: gar.front.ppi, shownAs: gar.colors[c] ? null : gar.fallback };
      return { silhouette: gar.back.silhouette, hex: hex, ref: gar.back.ref, ppi: gar.back.ppi };
    }
    var key = (gar.views[c] && gar.views[c][view]) ? c : (gar.colorMap[c] || Object.keys(gar.views)[0]);
    var v = gar.views[key] && gar.views[key][view];
    var shown = key;
    if (!v) { shown = 'Black'; v = gar.views.Black[view]; }
    var exact = (GARMENT === 'camo') || (shown === c);
    return { img: BASE + v.img, ref: v.ref, ppi: v.ppi, shownAs: exact ? null : shown };
  }

  /* ---------- silhouette backs for tees / crews (no back photos) ---------- */
  function shade(hex, f) {
    var n = parseInt(hex.slice(1), 16), r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
    return 'rgb(' + Math.round(r * f) + ',' + Math.round(g * f) + ',' + Math.round(b * f) + ')';
  }
  function drawSilhouette(ctx, kind, hex) {
    ctx.fillStyle = '#f3f1ee'; ctx.fillRect(0, 0, 1000, 1000);
    var p = new Path2D();
    if (kind === 'tee') {
      p.moveTo(430, 135); p.quadraticCurveTo(500, 158, 570, 135);
      p.lineTo(700, 172); p.lineTo(815, 318); p.lineTo(752, 392); p.lineTo(712, 345);
      p.lineTo(722, 742); p.lineTo(278, 742); p.lineTo(288, 345); p.lineTo(248, 392); p.lineTo(185, 318); p.lineTo(300, 172); p.closePath();
    } else {
      p.moveTo(432, 135); p.quadraticCurveTo(500, 156, 568, 135);
      p.lineTo(700, 175); p.lineTo(770, 330); p.lineTo(842, 690); p.lineTo(842, 722); p.lineTo(776, 722); p.lineTo(762, 690);
      p.lineTo(708, 420); p.lineTo(716, 742); p.lineTo(284, 742); p.lineTo(292, 420);
      p.lineTo(238, 690); p.lineTo(224, 722); p.lineTo(158, 722); p.lineTo(158, 690); p.lineTo(230, 330); p.lineTo(300, 175); p.closePath();
    }
    ctx.save(); ctx.shadowColor = 'rgba(0,0,0,.18)'; ctx.shadowBlur = 18; ctx.shadowOffsetY = 6;
    ctx.fillStyle = hex; ctx.fill(p); ctx.restore();
    ctx.save(); ctx.clip(p);
    var g = ctx.createLinearGradient(0, 120, 0, 760); g.addColorStop(0, 'rgba(255,255,255,.10)'); g.addColorStop(1, 'rgba(0,0,0,.12)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 1000, 1000);
    ctx.fillStyle = shade(hex, 0.86); ctx.fillRect(270, 720, 460, 26);          // hem band
    if (kind === 'crew') { ctx.fillRect(150, 700, 110, 30); ctx.fillRect(740, 700, 110, 30); }
    ctx.restore();
    ctx.lineWidth = 2; ctx.strokeStyle = shade(hex, 0.7); ctx.stroke(p);
    ctx.beginPath(); ctx.moveTo(432, 135); ctx.quadraticCurveTo(500, 156, 568, 135); ctx.lineWidth = 9; ctx.strokeStyle = shade(hex, 0.8); ctx.stroke();
  }

  /* ---------- placement math ---------- */
  function locBox(view, locId, base) {
    var L = cfg.locations[view][locId]; if (!L) return null;
    var top = (view === 'back' && gar.hoodBack && L.hoodTop != null) ? L.hoodTop : L.top;
    var ppi = base.ppi;
    return { cx: base.ref[0] + L.x * ppi, top: base.ref[1] + top * ppi, w: L.w * ppi, h: L.h * ppi, L: L, ppi: ppi };
  }
  function fit(box, iw, ih) {
    var k = Math.min(box.w / iw, box.h / ih);
    return { x: box.cx - iw * k / 2, y: box.top, w: iw * k, h: ih * k };
  }
  function designSrc(id) {
    var d = team.designs.filter(function (x) { return x.id === id; })[0];
    return d ? BASE + d.src : null;
  }

  /* ---------- render ---------- */
  var canvas, ctx, caption, sizeNote = { front: null, back: null }, warn = { front: null, back: null };
  var renderToken = 0, pendingWarn = { front: '', back: '' };
  function render() {
    var token = ++renderToken, view = state.view, base = baseFor(view);
    var jobs = [base.img ? loadImg(base.img) : Promise.resolve(null)];
    var s = state[view];
    var artPromise = Promise.resolve(null);
    if (s.design === 'upload' && s.upImg) artPromise = Promise.resolve(s.upImg);
    else if (s.design && s.design !== 'none' && s.design !== 'namenum' && s.design !== 'upload') artPromise = loadImg(designSrc(s.design));
    jobs.push(artPromise);
    Promise.all(jobs).then(function (r) {
      if (token !== renderToken) return;
      var photo = r[0], art = r[1];
      var W = photo ? photo.naturalWidth : 1000, H = photo ? photo.naturalHeight : 1000;
      canvas.width = W; canvas.height = H;
      if (photo) ctx.drawImage(photo, 0, 0); else drawSilhouette(ctx, base.silhouette, base.hex);
      var box = s.loc ? locBox(view, s.loc, base) : null;
      if (box && s.design !== 'none' && s.design !== 'namenum') {
        ctx.save(); ctx.setLineDash([10, 8]); ctx.lineWidth = Math.max(2, W / 400); ctx.strokeStyle = 'rgba(255,255,255,.55)';
        ctx.strokeRect(box.cx - box.w / 2, box.top, box.w, box.h);
        ctx.strokeStyle = 'rgba(0,0,0,.35)'; ctx.lineDashOffset = 9; ctx.strokeRect(box.cx - box.w / 2, box.top, box.w, box.h); ctx.restore();
      }
      if (art && box) {
        var f = fit(box, art.naturalWidth, art.naturalHeight);
        ctx.save(); ctx.globalAlpha = 0.97; ctx.drawImage(art, f.x, f.y, f.w, f.h); ctx.restore();
        var inW = f.w / box.ppi, inH = f.h / box.ppi;
        updateSize(view, box.L.label + ' \u2014 prints about ' + fmt(inW) + ' \u00d7 ' + fmt(inH) + ' in');
        if (s.design === 'upload') checkDpi(view, art, inW);
        else setWarn(view, '');
      } else if (s.design === 'namenum' && view === 'back') {
        drawNameNumber(base);
        updateSize('back', 'Last name about 2.5 in tall \u00b7 number about 8 in tall');
        setWarn('back', '');
      } else {
        updateSize(view, s.design === 'upload' ? 'Choose an image to see it on the ' + gar.label + '.' : '');
        setWarn(view, pendingWarn[view] || '');
      }
      caption.textContent = base.shownAs ? 'Preview shown on ' + base.shownAs + '. Your ' + gar.label + ' will be ' + state.color + '.' : '';
      syncFields();
    }).catch(function (e) { caption.textContent = 'Preview unavailable right now. Your choices are still saved with your order.'; syncFields(); });
  }
  var fontReady = null;
  function loadFont() {
    if (fontReady) return fontReady;
    try {
      var ff = new FontFace('OCBMVarsity', 'url(' + BASE + 'fonts/Graduate-Regular.ttf)');
      fontReady = ff.load().then(function (f) { document.fonts.add(f); return true; }).catch(function () { return false; });
    } catch (e) { fontReady = Promise.resolve(false); }
    return fontReady;
  }
  function drawNameNumber(base) {
    var s = state.back, c = team.nameColors || { fill: '#fff', key: '#000', edge: '#16489e' };
    var name = (s.name || 'LAST NAME').toUpperCase(), num = s.number || '00';
    var ppi = base.ppi, top = base.ref[1] + (gar.hoodBack ? 1.0 : 3.0) * ppi, cx = base.ref[0];
    function stack(txt, px, y, maxW) {
      ctx.font = px + 'px OCBMVarsity, "Arial Black", Impact, sans-serif';
      var w = ctx.measureText(txt).width, sx = Math.min(1, maxW / w);
      ctx.save(); ctx.translate(cx, y); ctx.scale(sx, 1); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.lineJoin = 'round';
      ctx.lineWidth = px * 0.22; ctx.strokeStyle = c.edge; ctx.strokeText(txt, 0, 0);
      ctx.lineWidth = px * 0.12; ctx.strokeStyle = c.key; ctx.strokeText(txt, 0, 0);
      ctx.fillStyle = c.fill; ctx.fillText(txt, 0, 0); ctx.restore();
    }
    stack(name, 2.5 * ppi * 1.35, top, 12 * ppi);
    stack(num, 8 * ppi * 1.35, top + 3.4 * ppi, 11 * ppi);
  }
  function updateSize(view, t) { if (sizeNote[view]) sizeNote[view].textContent = t; }
  function setWarn(view, t) { if (warn[view]) { warn[view].textContent = t; warn[view].style.display = t ? '' : 'none'; } }
  function checkDpi(view, img, inW) {
    var dpi = img.naturalWidth / inW;
    if (dpi < MIN_DPI_WARN) setWarn(view, 'This image is small (' + Math.round(dpi) + ' dpi at this size) and may print blurry. A larger file or a smaller spot will look sharper.');
    else if (dpi < MIN_DPI_OK) setWarn(view, 'Image quality is OK but not ideal (' + Math.round(dpi) + ' dpi at this size). We will clean it up before printing.');
    else setWarn(view, '');
  }

  /* ---------- order fields ---------- */
  function designLabel(id) {
    if (id === 'upload') return 'Customer upload';
    if (id === 'namenum') return 'Athlete last name and number';
    if (id === 'none' || !id) return 'None';
    var d = team.designs.filter(function (x) { return x.id === id; })[0]; return d ? d.label : id;
  }
  function placementText(view) {
    var s = state[view];
    if (!s.design || s.design === 'none' || s.design === 'namenum') return '';
    var L = cfg.locations[view][s.loc]; if (!L) return '';
    return L.label + ' (fits ' + L.w + ' \u00d7 ' + L.h + ' in)';
  }
  function syncFields() {
    ['front', 'back'].forEach(function (v) {
      var cap = v === 'front' ? 'Front' : 'Back', s = state[v];
      setField(cap + ' Design', designLabel(s.design));
      setField(cap + ' Placement', placementText(v));
      var fi = field(cap + ' Upload', 'file'); fi.disabled = s.design !== 'upload';
    });
    var nn = state.back.design === 'namenum';
    setField('Athlete Last Name', nn ? (state.back.name || '').toUpperCase() : '');
    setField('Athlete Number', nn ? state.back.number : '');
  }

  /* ---------- UI ---------- */
  var ui = {};
  function select(opts, value, onchange, id) {
    var s = el('select', { 'class': 'select__select', id: id, style: 'width:100%;padding:0 1rem;min-height:4.4rem', onchange: onchange });
    opts.forEach(function (o) { var op = el('option', { value: o[0], text: o[1] }); if (o[0] === value) op.selected = true; s.appendChild(op); });
    return s;
  }
  function sideBlock(view) {
    var s = state[view], cap = view === 'front' ? 'Front' : 'Back';
    var designOpts = team.designs.map(function (d) { return [d.id, d.label]; });
    if (view === 'back') designOpts.push(['namenum', 'Athlete last name and number']);
    designOpts.push(['upload', 'Upload my own image']); designOpts.push(['none', 'Nothing on the ' + view]);
    var locOpts = Object.keys(cfg.locations[view]).map(function (k) { var L = cfg.locations[view][k]; return [k, L.label + ' \u2014 up to ' + L.w + ' \u00d7 ' + L.h + ' in']; });
    var upBtn = el('button', { type: 'button', 'class': 'button button--secondary', style: 'margin:.8rem 0 0;width:100%;min-height:4.4rem',
      onclick: function () { field(cap + ' Upload', 'file').click(); } }, ['Choose image']);
    var upName = el('p', { style: 'margin:.4rem 0 0;font-size:1.2rem;opacity:.75' });
    var upWrap = el('div', { style: s.design === 'upload' ? '' : 'display:none' }, [upBtn, upName,
      el('p', { style: 'margin:.4rem 0 0;font-size:1.15rem;opacity:.7', text: 'PNG or JPG. A transparent PNG looks best. We size it to fit the spot you pick and check it before printing.' })]);
    var locSel = select(locOpts, s.loc, function (e) { s.loc = e.target.value; state.view = view; refreshTabs(); render(); }, 'ocbm-' + view + '-loc');
    var locWrap = el('div', { style: (s.design === 'none' || s.design === 'namenum') ? 'display:none' : '' }, [
      el('label', { 'class': 'form__label', 'for': 'ocbm-' + view + '-loc', text: cap + ' location' }), el('div', { 'class': 'select' }, [locSel])]);
    var nameIn = el('input', { 'class': 'field__input', type: 'text', maxlength: '16', placeholder: 'e.g. SMITH', autocapitalize: 'characters',
      style: 'text-transform:uppercase;margin-bottom:.8rem', value: s.name || '', oninput: function (e) { s.name = e.target.value; render(); } });
    var numIn = el('input', { 'class': 'field__input', type: 'text', inputmode: 'numeric', pattern: '[0-9]{1,2}', maxlength: '2', placeholder: 'e.g. 12',
      value: s.number || '', oninput: function (e) { s.number = e.target.value.replace(/\D/g, ''); e.target.value = s.number; render(); } });
    var nnWrap = view === 'back' ? el('div', { style: s.design === 'namenum' ? '' : 'display:none' }, [
      el('label', { 'class': 'form__label', text: 'Athlete last name' }), nameIn, el('label', { 'class': 'form__label', text: 'Athlete number' }), numIn]) : null;
    var desSel = select(designOpts, s.design, function (e) {
      s.design = e.target.value; state.view = view; pendingWarn[view] = '';
      upWrap.style.display = s.design === 'upload' ? '' : 'none';
      locWrap.style.display = (s.design === 'none' || s.design === 'namenum') ? 'none' : '';
      if (nnWrap) nnWrap.style.display = s.design === 'namenum' ? '' : 'none';
      refreshTabs(); render();
    }, 'ocbm-' + view + '-design');
    var fi = field(cap + ' Upload', 'file');
    fi.onchange = function () {
      var f = fi.files && fi.files[0]; if (!f) return;
      if (f.size > 20 * 1024 * 1024) { upName.textContent = 'That file is over 20 MB. Please pick a smaller one.'; fi.value = ''; return; }
      var url = URL.createObjectURL(f), im = new Image();
      im.onload = function () { pendingWarn[view] = ''; s.upImg = im; s.file = f.name; upName.textContent = f.name + ' \u00b7 ' + im.naturalWidth + ' \u00d7 ' + im.naturalHeight + ' px'; state.view = view; refreshTabs(); render(); };
      im.onerror = function () { upName.textContent = 'That file could not be read. Please use a PNG or JPG.'; fi.value = ''; };
      im.src = url;
    };
    if (s.file) upName.textContent = s.file;
    sizeNote[view] = el('p', { style: 'margin:.5rem 0 0;font-size:1.2rem;opacity:.8' });
    warn[view] = el('p', { style: 'display:none;margin:.5rem 0 0;font-size:1.2rem;color:#b3261e' });
    return el('div', { 'class': 'ocbm-side', style: 'margin-top:1.2rem' }, [
      el('p', { style: 'margin:0 0 .4rem;font-weight:600', text: cap }),
      el('label', { 'class': 'form__label', 'for': 'ocbm-' + view + '-design', text: cap + ' design' }), el('div', { 'class': 'select', style: 'margin-bottom:.8rem' }, [desSel]),
      locWrap, upWrap, nnWrap, sizeNote[view], warn[view]]);
  }
  function refreshTabs() {
    ['front', 'back'].forEach(function (v) {
      var b = ui['tab_' + v]; if (!b) return;
      var on = state.view === v; b.setAttribute('aria-pressed', on); b.style.background = on ? 'rgb(var(--color-foreground))' : 'transparent'; b.style.color = on ? 'rgb(var(--color-background))' : 'rgb(var(--color-foreground))';
    });
  }
  function build() {
    mount.innerHTML = '';
    canvas = el('canvas', { style: 'width:100%;height:auto;display:block;border-radius:8px;background:#eee', 'aria-label': 'Live preview of your design' });
    ctx = canvas.getContext('2d');
    caption = el('p', { style: 'margin:.4rem 0 0;font-size:1.15rem;opacity:.7;min-height:1em' });
    function tab(v, label) {
      var b = el('button', { type: 'button', id: 'ocbm-tab-' + v, style: 'flex:1;min-height:4rem;border:1px solid rgb(var(--color-foreground));cursor:pointer;font:inherit',
        onclick: function () { state.view = v; refreshTabs(); render(); } }, [label]);
      ui['tab_' + v] = b; return b;
    }
    mount.appendChild(el('div', { 'class': 'ocbm-designer', style: 'margin:0 0 1.6rem;padding:1.2rem;border:1px solid rgba(var(--color-foreground),.15);border-radius:10px' }, [
      el('p', { style: 'margin:0 0 .2rem;font-weight:700;font-size:1.6rem', text: 'Design it' }),
      el('p', { style: 'margin:0 0 1rem;font-size:1.25rem;opacity:.75', text: 'Pick what goes on the front and back and see it right away. Included free.' }),
      el('div', { style: 'display:flex;margin-bottom:.8rem;border-radius:6px;overflow:hidden' }, [tab('front', 'Front'), tab('back', 'Back')]),
      canvas, caption, sideBlock('front'), sideBlock('back'),
      el('p', { style: 'margin:1rem 0 0;font-size:1.1rem;opacity:.7', text: 'Preview is approximate. We confirm size and placement before printing, and we will reach out if an uploaded image needs work.' })]));
    refreshTabs();
  }

  /* ---------- block add-to-cart when an upload was chosen but no file picked ---------- */
  function guard(e) {
    var form = e.target; if (!form || form.id !== FORM_ID) return;
    var miss = ['front', 'back'].filter(function (v) { return state[v].design === 'upload' && !state[v].upImg; });
    if (miss.length) {
      e.preventDefault(); e.stopImmediatePropagation();
      pendingWarn[miss[0]] = 'Please choose an image to upload, or pick a different ' + miss[0] + ' design.';
      state.view = miss[0]; refreshTabs(); render(); setWarn(miss[0], pendingWarn[miss[0]]);
      mount.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
  document.addEventListener('submit', guard, true);

  /* ---------- color changes + section re-renders ---------- */
  function watch() {
    setInterval(function () {
      var c = readColor();
      if (!document.getElementById('ocbm-designer')) return;
      var m = document.getElementById('ocbm-designer');
      if (m !== mount || !m.firstChild) { mount = m; build(); render(); }
      if (c && c !== state.color) { state.color = c; render(); }
    }, 600);
    document.addEventListener('change', function (e) {
      if (e.target && e.target.name && e.target.name.indexOf('Color') !== -1) { state.color = readColor() || state.color; render(); }
    });
  }

  /* ---------- start ---------- */
  fetch(BASE + 'config.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(function (c) {
    cfg = c; team = cfg.teams[TEAM]; gar = cfg.garments[GARMENT];
    if (!team || !gar) { mount.style.display = 'none'; return; }
    state.color = readColor() || (gar.fallback || Object.keys(gar.colorMap || gar.colors || {})[0]);
    state.front.design = team.front.design;
    state.front.loc = (team.front.locByGarment && team.front.locByGarment[GARMENT]) || team.front.loc;
    state.back.design = team.back.design; state.back.loc = team.back.loc;
    loadFont().then(function () { if (state.back.design === 'namenum') render(); });
    build(); render(); watch();
  }).catch(function () { mount.style.display = 'none'; });
})();
