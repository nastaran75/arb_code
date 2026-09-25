r"""Single source of truth for the look of every figure that goes into the paper.

The paper's style gives
\textwidth = 5.5in and a 10pt Times body (\small 9pt, \footnotesize 8pt,
\scriptsize 7pt).  Two rules make a matplotlib figure look like it belongs there:

  1. DRAW AT FINAL SIZE.  A figure drawn 8.2in wide and included at
     width=\textwidth is squeezed to 0.67x on the page, and every font goes with
     it -- an 8.5pt tick label lands at 5.7pt, which is what makes a plot read as
     "screenshot someone shrank" rather than as part of the document.  Use
     width(frac) for the true inch width of the \includegraphics fraction the
     figure is actually included at, and never let LaTeX rescale.

  2. MATCH THE TEXT.  Times body -> serif text and STIX math (STIX Two is
     Times-metric-compatible, so this needs no LaTeX run), plus Type-42 embedded
     fonts instead of matplotlib's default Type 3.

Because of rule 1, FS is in TRUE POINTS: FS["tick"] == 7.5 renders at 7.5pt on
the printed page, directly comparable to the 10pt body text.

Importing this module applies the rcParams; every figure script should import it
before creating a figure.  plots/fig1.py re-exports FS/SC/INK/GRID/AXIS, so the
scripts that already read them off `fig1` keep working unchanged.
"""
import matplotlib

# ---- paper geometry -------------------------------------------------------------------------------------------
TEXTWIDTH = 5.5                          # \textwidth of the paper, in inches
BODY_PT = 10.0                           # \normalsize of the paper; FS below is relative to this


def width(frac=1.0):
    """Inch width to draw at, for a figure included at width=<frac>\\textwidth."""
    return TEXTWIDTH * frac


# ---- type -----------------------------------------------------------------------------------------------------
# True points on the page.  Roughly \small (9) for titles/labels and \scriptsize (7)
# for ticks and legends -- the usual relationship between a caption'd figure and body text.
FS = {"title": 9.0, "label": 8.5, "tick": 7.0, "legend": 7.0, "annot": 8.0,
      "dense": 6.5}                     # "dense": crowded categorical tick labels (27 model names on one axis)
SC = 1.0                                 # global font/stroke multiplier (make_panel.py --figwidth sets it)

# ---- ink ------------------------------------------------------------------------------------------------------
INK, GRID, AXIS = "#0b0b0b", "#e1e0d9", "#8f8e89"
W_PAD = 0.4

# Times New Roman first: it is the closest match to the paper's \usepackage{times} AND it is
# the only candidate here shipping a real bold face (STIX Two Text is regular/italic only, so
# fontweight="bold" silently degrades to regular under it).
SERIF = ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"]

RC = {
    # Times-matching text and math, so \usepackage{times} body text and the figure
    # labels share a look without paying for text.usetex on every draw.
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",          # matplotlib's bundled STIX: the Times-companion math set

    # Defaults for anything a script does not size explicitly.
    "font.size": FS["label"],
    "axes.titlesize": FS["title"],
    "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"],
    "figure.titlesize": FS["title"],

    # Hairlines: at 1.0x scale matplotlib's defaults are heavy next to 10pt Times.
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.3,

    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",

    # Embed real fonts.  matplotlib's default (Type 3) is rejected by some camera-ready
    # checkers and renders badly in several PDF viewers.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

matplotlib.rcParams.update(RC)


def style_axes(ax, grid_axis="both"):
    """The house axis treatment: no top/right spine, hairline grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5 * SC, ls="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5 * SC, width=0.5, labelsize=FS["tick"] * SC, colors=AXIS, labelcolor=INK)


def save(fig, out, dpi=200, tight=True):
    """Write <out>.pdf and <out>.png.

    bbox_inches="tight" trims to the drawn content, so the saved file is narrower
    than figsize and LaTeX scales it back up at width=\\textwidth -- which undoes
    rule 1 above.  Pass tight=False for a figure whose on-page point sizes have to
    be exact; the constrained/tight layout should already have removed the margin.
    """
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(out + ".pdf", metadata={"CreationDate": None}, **kw)   # no timestamp -> byte-reproducible
    fig.savefig(out + ".png", dpi=dpi, metadata={"Software": None}, **kw)
