"""Shared figure settings for the Nature Astronomy final artwork (revision 3).

Nature artwork guide: sans-serif (Arial/Helvetica) text of 5-7 pt at final
size, lines >= 0.25 pt, double-column width 180 mm, single-column 88 mm, and
vector PDFs with *editable* text (TrueType fonts, not matplotlib's default
Type 3 glyphs).
"""
import os
from pathlib import Path
import matplotlib as mpl

MM = 1 / 25.4                     # inches per mm
DOUBLE_COL = 180 * MM             # 180 mm
SINGLE_COL = 88 * MM              # 88 mm

# all figures are written to <repo>/outputs/figures
OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# font sizes in points at final print size
FS_TINY = 5.0      # dense annotations (SN names, spectral line labels)
FS_SMALL = 5.5     # legends, tick labels
FS = 6.5           # axis labels, in-panel text
FS_PANEL = 8.0     # bold panel letters (a, b, c)


def apply():
    mpl.rcParams.update({
        'pdf.fonttype': 42,             # TrueType -> editable text in Illustrator
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'mathtext.sf': 'Arial',
        'mathtext.fallback': 'cm',
        'font.size': FS,
        'axes.labelsize': FS,
        'axes.titlesize': FS,
        'xtick.labelsize': FS_SMALL,
        'ytick.labelsize': FS_SMALL,
        'legend.fontsize': FS_SMALL,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'lines.linewidth': 0.9,
        'lines.markersize': 3,
        'errorbar.capsize': 0,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'savefig.dpi': 300,
    })


def save(fig, name, **kw):
    """Save <name>.pdf (vector, editable text) and a <name>.png preview."""
    kw.setdefault('bbox_inches', 'tight')
    kw.setdefault('pad_inches', 0.02)
    pdf = OUT_DIR / f'{name}.pdf'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, **kw)
    fig.savefig(OUT_DIR / f'{name}.png', **{**kw, 'dpi': 200})
    print(f'saved {pdf}')
    return pdf
