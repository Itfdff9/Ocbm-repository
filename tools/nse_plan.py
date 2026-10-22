"""Plan for the 4 NSE designs (2026-10-21): own listings in Core-7 colors, front print only."""
import json, os
R = '/home/claude/ocbm-repository'
D2 = '/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/drive2048'
DESIGNS = {  # slug: (title, print png, tag, adults_only)
 'nse-cheer-dad':        ('Cheer Dad',        f'{R}/designs/nse-cheer-dad/print/NSE_CheerDad_v2_14in_300dpi.png', 'Design: Cheer Dad', True),
 'nse-cheer-bow':        ('Cheer Bow',        f'{R}/designs/nse-cheer-bow/print/NSE_CheerBow_NorthShoreElite_14in_300dpi.png', 'Design: Cheer Bow', False),
 'nse-star-logo':        ('Star Logo',        f'{R}/designs/nse-star-logo/print/NSE_StarLogo_14in_300dpi.png', 'Design: Star Logo', False),
 'nse-blue-black-white': ('Blue Black White', f'{R}/designs/nse-blue-black-white/print/NSE_BlueBlackWhite_14in_300dpi.png', 'Design: Blue Black White', False),
}
# blank: list of (Shopify color name, kind, args) — default (with models)
G5000 = [(c, 'tee', (f'{D2}/g5000/{f}.jpg',)) for c, f in [('Black', 'Black'), ('White', 'White'), ('Royal', 'Royal'), ('Ash', 'Ash-Grey'), ('Charcoal', 'Charcoal'), ('Light Pink', 'Light-Pink'), ('Heliconia', 'Heliconia')]]
G18000 = [(c, 'crew', (f'{D2}/g18000/{f}.jpg',)) for c, f in [('Black', 'Black'), ('White', 'White'), ('Royal', 'Royal-Blue'), ('Ash', 'Ash-Grey'), ('Light Pink', 'Light-Pink'), ('Heliconia', 'Heliconia')]]
G18000.insert(4, ('Charcoal', 'crew', (f'{R}/customizer/blanks/crew/Charcoal.jpg',)))  # Drive 'Charcoal (3)' is a bluish photo; repo 1000 px Charcoal is the true color
G18500 = [('Black', 'hoodie_model', ('Black',)), ('White', 'flat', ('hoodie_18500', 'White')), ('Royal', 'flat', ('hoodie_18500', 'Royal')),
          ('Ash', 'hoodie_model', ('Ash',)), ('Charcoal', 'flat', ('hoodie_18500', 'Charcoal')), ('Light Pink', 'flat', ('hoodie_18500', 'Light Pink')),
          ('Heliconia', 'flat', ('hoodie_18500', 'Heliconia'))]
BC3001 = [('Black', 'life', ('seated', 'Black')), ('White', 'life', ('couch', 'White')), ('True Royal', 'flat', ('bc_3001', 'True Royal')),
          ('Ash', 'life', ('seated', 'Ash')), ('Asphalt', 'life', ('seated', 'Asphalt')), ('Soft Pink', 'life', ('seated', 'Soft-Pink')),
          ('Fuchsia', 'life', ('seated', 'Fuchsia'))]
CC1717 = [(c, 'life', ('flatlay', f)) for c, f in [('Black', 'Black'), ('White', 'White'), ('Grey', 'Grey'), ('Pepper', 'Pepper'), ('Blossom', 'Blossom'), ('Heliconia', 'Heliconia')]]
YOUTH = [(c, 'flat', ('youth_tee_5000b', c)) for c in ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Light Pink', 'Heliconia']]
TODDLER = [(c, 'flat', ('toddler_tee_3301t', c)) for c in ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Pink', 'Hot Pink']]
ONESIE = [(c, 'flat', ('onesie_4400', c)) for c in ['Black', 'White', 'Royal', 'Heather', 'Charcoal', 'Pink', 'Hot Pink']]
# Cheer Dad: no female models -> S&S flats on crew, hoodie, B+C
FLATS_DAD = {
 'g18000': [(c, 'flat', ('crew_18000', c)) for c in ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Light Pink', 'Heliconia']],
 'g18500': [(c, 'flat', ('hoodie_18500', c)) for c in ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Light Pink', 'Heliconia']],
 'bc3001': [(c, 'flat', ('bc_3001', c)) for c in ['Black', 'White', 'True Royal', 'Ash', 'Asphalt', 'Soft Pink', 'Fuchsia']],
}
BLANKS = {'g5000': G5000, 'bc3001': BC3001, 'cc1717': CC1717, 'g18000': G18000, 'g18500': G18500,
          'youth_5000b': YOUTH, 'toddler_3301t': TODDLER, 'onesie_4400': ONESIE}
KIDS = ('youth_5000b', 'toddler_3301t', 'onesie_4400')

def plan():
    jobs, P = [], {}
    for slug, (title, png, tag, adults_only) in DESIGNS.items():
        for blank, rows in BLANKS.items():
            if adults_only and blank in KIDS: continue
            if slug == 'nse-cheer-dad' and blank in FLATS_DAD: rows = FLATS_DAD[blank]
            for color, kind, args in rows:
                out = f"{R}/designs/{slug}/{blank}/{color.replace(' ', '-')}_Front.jpg"
                jobs.append([png, kind, list(args), out])
                P.setdefault(slug, {}).setdefault(blank, {})[color] = out
    return jobs, P

if __name__ == '__main__':
    jobs, P = plan()
    json.dump({'jobs': jobs}, open('/home/claude/work/nse/jobs.json', 'w'))
    json.dump(P, open(f'{R}/designs/nse_plan.json', 'w'), indent=0)
    print(len(jobs), 'jobs;', {k: sum(len(v) for v in d.values()) for k, d in P.items()})
