"""The paper's revenue-spectrum heatmap figures (main Figure 3 and its appendix twin).

Each figure is one row of heatmaps (fig_rev_heat.draw): rows = base models sorted by
per-attempt price, x = deployment budget per task, cell = share of the arbitrageur's
spend. The frontier figures they sit next to (main Figure 2 / appendix) are rendered and
mirrored by plots/fig1.py (fig1.pdf / fig1_appendix.pdf), so the main text and the appendix
show the same two ADJACENT figures for their respective pair of markets.

MAIN     = LiveCodeBench + Terminal-Bench 2.0 -> figures/fig_rev_main.pdf
APPENDIX = Terminal-Bench 4.0 + DeepSWE       -> figures/fig_rev_appendix.pdf

A two-row composite (frontier row above the heatmaps, `heat_only=False`) is still supported
by `compose` -- set `top_band` and drop `heat_only` in FIGS to get it back.

Layout is explicit (no tight_layout): the heatmaps carry ~1in of model-name labels on the
left and get a FIXED physical cell height so markets with different numbers of funded
models read at the same row thickness.

    ../.venv/bin/python fig_appendix.py            # both figures
    ../.venv/bin/python fig_appendix.py main       # just one
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
import fig1 as F                            # noqa: E402
import fig_rev_heat as RH                   # noqa: E402

PAPER_DIR = os.path.join(HERE, "..", "figures")
CELL_IN = 0.115                             # heatmap row height, inches per model row
LEGEND_RAISE_IN = 0.18                      # legend offset above the panels, inches (as in fig1)

# Per-figure geometry: bands are (bottom, top) in figure fraction. The heatmaps hang from
# the top of their band at fixed cell height (anchor N + box aspect), so bot_band's top is
# what matters; fig_h leaves room for the tallest heatmap (LiveCodeBench keeps 15 rows).
FIGS = {
    "main": dict(keys=["livecodebench_priced", "terminal_bench2_priced"],
                 titles=("LiveCodeBench", "Terminal-Bench 2.0"),
                 # bot_wspace holds the RIGHT heatmap's row labels, which are now
                 # "Agent x Model" and about twice as wide as a bare model name -- at 0.50
                 # they ran back over the LiveCodeBench cells. bot_left shrinks to buy the
                 # extra gap back without narrowing either heatmap.
                 heat_only=True, fig_h=2.4, bot_band=(0.05, 0.90),
                 bot_left=0.155, bot_right=0.93, bot_wspace=1.05,
                 # One row per SYSTEM (Agent x Model), not per base model: the allocation is
                 # solved per system, and five Terminal-Bench systems share a base model but
                 # do NOT share its revenue (only 2 of the 5 GPT-5.3-Codex scaffolds ever earn).
                 # Pooling them would credit the model for a scaffold's result. No effect on
                 # LiveCodeBench, where no two systems share a base model.
                 group=False,
                 out="fig_rev_main", paper="fig_rev_main.pdf"),
    "appendix": dict(keys=["terminal_bench4_priced", "deepswe_priced"],
                     titles=("Terminal-Bench 4.0", "DeepSWE"),
                     heat_only=True, fig_h=2.4, bot_band=(0.05, 0.90),
                     # same label gap as the main figure: per-configuration row labels are wide
                     bot_left=0.155, bot_right=0.93, bot_wspace=1.05,
                     # One row per configuration, as in the main figure. On DeepSWE the rows are
                     # then labelled with the reasoning-effort level, which shows that every funded
                     # configuration is its model's maximum effort; pooling by model hid that.
                     group=False,
                     out="fig_rev_appendix", paper="fig_rev_appendix.pdf"),
}


def compose(cfg):
    heat_only = cfg.get("heat_only", False)
    fig_w, fig_h = F.PS.width(1.0), cfg["fig_h"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    if not heat_only:
        items = [F.load(os.path.join(F.DATA_DIR, f"{k}.json")) for k in cfg["keys"]]
        gs_top = fig.add_gridspec(1, 3, left=0.075, right=0.985,
                                  bottom=cfg["top_band"][0], top=cfg["top_band"][1], wspace=0.55)
        axs_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
        handles = []
        for ax, d in zip(axs_top, items):
            handles = F.draw_cost(ax, d) or handles
        profit_handles = F.draw_profit(axs_top[2], items)
    gs_bot = fig.add_gridspec(1, 2, left=cfg["bot_left"], right=cfg["bot_right"],
                              bottom=cfg["bot_band"][0], top=cfg["bot_band"][1],
                              wspace=cfg["bot_wspace"])   # the gap holds the right heatmap's model-name labels
    axs_bot = [fig.add_subplot(gs_bot[0, i]) for i in range(2)]
    for ax, key, title in zip(axs_bot, cfg["keys"], cfg["titles"]):
        pc = RH.draw(key, ax, prices=False, group=cfg.get("group", True))     # rows stay price-sorted; the numbers are caption material
        ax.set_title(title, fontsize=F.PS.FS["title"], pad=4)   # without the "(N systems)" suffix
        # the "deployment"/"(CV)" qualifiers of the standalone version are caption material here
        ax.set_xlabel("Budget per task ($)", fontsize=F.PS.FS["label"])
        # fixed physical cell height, rows hanging from the top of the band
        n_rows = len(ax.get_yticks())
        w_in = (ax.get_position().x1 - ax.get_position().x0) * fig_w
        ax.set_box_aspect(n_rows * CELL_IN / w_in)
        ax.set_anchor("N")
    fig.canvas.draw()                                       # realize box aspects before measuring
    # colourbar height follows the TALLER heatmap (more rows), not whichever panel is rightmost
    pos = max((ax.get_position() for ax in axs_bot), key=lambda p: p.height)
    cax = fig.add_axes([gs_bot.right + 0.03, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("Revenue share (%)", fontsize=F.PS.FS["label"])
    cb.ax.tick_params(labelsize=F.PS.FS["tick"])
    if heat_only:
        return _save(fig, cfg)
    # legends as in fig1.draw_fig1
    dy = LEGEND_RAISE_IN / fig_h
    cost_axs = axs_top[:2]
    x0 = cost_axs[0].get_position().x0, cost_axs[-1].get_position().x1
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(sum(x0) / 2, cost_axs[0].get_position().y1 + dy),
               ncol=len(handles), fontsize=F.FS["legend"], frameon=False,
               handlelength=2.2, columnspacing=1.6, labelcolor=F.INK)
    ppos = axs_top[2].get_position()
    leg = fig.legend(handles=profit_handles, loc="lower center",
                     bbox_to_anchor=((ppos.x0 + ppos.x1) / 2, ppos.y1 + dy),
                     ncol=2, fontsize=F.FS["legend"], frameon=False,
                     handlelength=1.6, columnspacing=1.0, handletextpad=0.5, labelcolor=F.INK)
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    if lb.x1 > ppos.x1:
        leg.set_bbox_to_anchor(((ppos.x0 + ppos.x1) / 2 - (lb.x1 - ppos.x1), ppos.y1 + dy))
    _save(fig, cfg)


def _save(fig, cfg):
    out = os.path.join(HERE, cfg["out"])
    fig.savefig(out + ".pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", out + ".pdf")
    if os.path.isdir(PAPER_DIR):
        shutil.copy(out + ".pdf", os.path.join(PAPER_DIR, cfg["paper"]))
        print("copied to", os.path.normpath(os.path.join(PAPER_DIR, cfg["paper"])))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for name in which:
        compose(FIGS[name])
