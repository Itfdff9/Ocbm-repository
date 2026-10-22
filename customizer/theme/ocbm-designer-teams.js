/* Built-in team settings for teams not yet in customizer/config.json. Loaded before ocbm-designer.js.
   back.swap: true -> the back shows the team design by default, and a checkbox lets the customer
   print the player's last name and number on the back instead (no extra charge). */
window.OCBM_TEAMS = {
  DREAMS: {
    designs: [{ id: 'dreams_logo', label: 'DREAMS Fastpitch logo', src: '../dreams/logo/DREAMS_PARENT_cleaned_300dpi.png' }],
    front: { design: 'dreams_logo' },
    back: { design: 'namenum' },
    nameColors: { fill: '#ffffff', key: '#000000', edge: '#16489e' }
  },
  BUFC: {
    designs: [{ id: 'bufc_crest', label: 'Bethpage United FC crest', src: '../bufc/logo/BUFC_crest_transparent.png' }],
    front: { design: 'bufc_crest' },
    back: { design: 'namenum' },
    nameColors: { fill: '#0e1c46', key: '#f8f8f6', edge: '#0e1c46' }
  },
  BLUEBIRDS: {
    designs: [{ id: 'bluebirds_crest', label: 'Bethpage Bluebirds FC crest', src: '../bluebirds/logo/BLUEBIRDS_crest_transparent.png' }],
    front: { design: 'bluebirds_crest' },
    back: { design: 'namenum' },
    nameColors: { fill: '#ffffff', key: '#0a1e52', edge: '#c0aa76' }
  },
  BLUECRUSH: {
    designs: [{ id: 'bluecrush_logo', label: 'Bethpage Blue Crush logo', src: '../designs/bethpage-blue-crush/logo/BLUECRUSH_logo_1200.png' }],
    front: { design: 'bluecrush_logo' },
    back: { design: 'bluecrush_logo', loc: 'full_back', swap: true,
            swapLabel: "Print my player's last name and number on the back instead of the logo" },
    nameColors: { fill: '#ffffff', key: '#0b1d4d', edge: '#1e6fd9' }
  }
};

/* North Shore Elite single-design listings (tag "Design: …" on the product): same NSE design menu,
   but the front starts on that listing's design and the back starts blank. */
(function () {
  var NSE_DESIGNS = [
    { id: 'nse', label: 'North Shore Elite emblem', src: 'designs/nse_emblem.png' },
    { id: 'bbw', label: 'Blue Black White', src: 'designs/blue_black_white.png' },
    { id: 'star_logo', label: 'NS North Shore Elite star logo', src: 'designs/nse_star_logo.png' },
    { id: 'cheer_bow', label: 'North Shore Elite cheer bow', src: 'designs/nse_cheer_bow.png' },
    { id: 'cheer_dad', label: 'Cheer Dad (Loud · Proud · Always There)', src: 'designs/nse_cheer_dad.png' }
  ];
  var map = { NSE_CHEERDAD: 'cheer_dad', NSE_CHEERBOW: 'cheer_bow', NSE_STARLOGO: 'star_logo', NSE_BBW: 'bbw' };
  Object.keys(map).forEach(function (k) {
    window.OCBM_TEAMS[k] = {
      designs: NSE_DESIGNS,
      front: { design: map[k], loc: 'center_chest' },
      back: { design: 'none', loc: 'full_back' },
      nameColors: { fill: '#f8f8f6', key: '#0c0c0c', edge: '#16489e' }
    };
  });
})();

/* Back swap checkbox: sits at the top of the designer's Back section and drives its Back design menu,
   so the order notes ("Back Design", "Athlete Last Name", "Athlete Number") stay exactly as the designer writes them. */
(function () {
  'use strict';
  function teamCfg() {
    var m = document.getElementById('ocbm-designer');
    var t = m && window.OCBM_TEAMS[m.getAttribute('data-team')];
    return t && t.back && t.back.swap ? t : null;
  }
  function attach() {
    var t = teamCfg(); if (!t) return;
    var sel = document.getElementById('ocbm-back-design');
    if (!sel || sel.getAttribute('data-swap') === '1') return;
    sel.setAttribute('data-swap', '1');
    var side = sel.closest('.ocbm-side'); if (!side) return;
    var cb = document.createElement('input');
    cb.type = 'checkbox'; cb.id = 'ocbm-back-swap';
    cb.style.cssText = 'width:2.2rem;height:2.2rem;margin:.1rem .9rem 0 0;flex:none';
    cb.checked = sel.value === 'namenum';
    var lab = document.createElement('label');
    lab.setAttribute('for', 'ocbm-back-swap');
    lab.style.cssText = 'display:flex;align-items:flex-start;margin:0 0 1rem;padding:1rem 1.2rem;border-radius:8px;border:1px solid rgba(var(--color-foreground),.25);font-size:1.4rem;line-height:1.4;cursor:pointer';
    var txt = document.createElement('span');
    txt.innerHTML = '<strong>' + t.back.swapLabel + '</strong><br><span style="font-size:1.2rem;opacity:.75">Included free. Leave unchecked to keep the logo on the back.</span>';
    lab.appendChild(cb); lab.appendChild(txt);
    var head = side.firstChild;
    side.insertBefore(lab, head ? head.nextSibling : null);
    cb.addEventListener('change', function () {
      sel.value = cb.checked ? 'namenum' : t.back.design;
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      if (cb.checked) {
        side.querySelectorAll('input[type=checkbox]').forEach(function (x) {
          if (x !== cb && !x.checked) { x.checked = true; x.dispatchEvent(new Event('change', { bubbles: true })); }
        });
        var tab = document.getElementById('ocbm-tab-back'); if (tab) tab.click();
      }
    });
    sel.addEventListener('change', function () { cb.checked = sel.value === 'namenum'; });
  }
  if (!teamCfg()) return;
  setInterval(attach, 500);
})();
