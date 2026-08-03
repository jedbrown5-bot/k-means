"""
SPA317 / SPA442  Week 9  The Clustering Loop, step by step.

An interactive visualiser for the loop at the heart of ISODATA. Students place
centres, then step through the two moves one at a time:
  1. Assign  every point to its nearest centre  (nearest centre wins)
  2. Update  each centre to the middle of its points
and repeat until nothing changes (convergence). An ISODATA mode adds the split
and merge moves, so the number of clusters adjusts itself.

Run it with:   streamlit run Week9_kmeans_game.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- house palette (matches the Week 9 lecture figures) --------------------
PAPER = "#fbfaf7"; INK = "#1a1a1a"; GREY = "#9aa0a8"
WATER = "#1d4e89"; FOREST = "#2e7d32"; PASTURE = "#7cb342"; BARE = "#e08a1e"
CLASS_COLS = [WATER, FOREST, PASTURE, BARE]
CLASS_NAMES = ["water", "forest", "pasture and crop", "bare and built"]
# distinct colours for arbitrary clusters
CLUSTER_COLS = [WATER, FOREST, BARE, "#7b3fa0", PASTURE, "#c0392b", "#00838f",
                "#8d6e63", "#d81b60", "#3949ab", "#f9a825", "#546e7a",
                "#26a69a", "#ad1457", "#5d4037", "#1565c0"]

# ============================================================================
# Pure clustering logic (no Streamlit here, so it can be tested on its own)
# ============================================================================
def make_points(dataset, n, seed):
    """Return (X, truth): n points in 2D and their true-cover labels."""
    rng = np.random.default_rng(seed)
    if dataset == "Coffs feature space (4 covers)":
        specs = [((0.06, 0.09), (0.020, 0.020)),   # water
                 ((0.12, 0.42), (0.030, 0.055)),   # forest
                 ((0.30, 0.34), (0.045, 0.040)),   # pasture and crop
                 ((0.40, 0.16), (0.038, 0.032))]   # bare and built
        xlab, ylab, lim = "Red", "NIR", (0, 0.50)
    elif dataset == "Three clean blobs":
        specs = [((0.22, 0.24), (0.045, 0.045)),
                 ((0.32, 0.72), (0.050, 0.050)),
                 ((0.72, 0.40), (0.055, 0.055))]
        xlab, ylab, lim = "feature 1", "feature 2", (0, 0.95)
    else:  # Overlapping / tricky
        specs = [((0.35, 0.45), (0.11, 0.030)),    # elongated
                 ((0.45, 0.62), (0.055, 0.055)),   # overlaps the next
                 ((0.55, 0.60), (0.055, 0.055)),
                 ((0.72, 0.28), (0.050, 0.050))]
        xlab, ylab, lim = "feature 1", "feature 2", (0, 1.0)
    per = max(1, n // len(specs))
    X, truth = [], []
    for k, ((cx, cy), (sx, sy)) in enumerate(specs):
        pts = np.column_stack([rng.normal(cx, sx, per), rng.normal(cy, sy, per)])
        X.append(pts); truth += [k] * per
    X = np.clip(np.vstack(X), lim[0], lim[1])
    return X, np.array(truth), (xlab, ylab, lim)

def init_centres(X, k, seed):
    """k-means++ style seeding: spread the initial centres out a little."""
    rng = np.random.default_rng(seed + 999)
    idx = [rng.integers(len(X))]
    for _ in range(1, k):
        d = np.min([((X - X[i]) ** 2).sum(1) for i in idx], axis=0)
        p = d / (d.sum() + 1e-12)
        idx.append(rng.choice(len(X), p=p))
    return X[idx].astype(float).copy()

def assign(X, centres):
    return (((X[:, None, :] - centres[None, :, :]) ** 2).sum(2)).argmin(1)

def update(X, labels, centres):
    new = centres.copy()
    for j in range(len(centres)):
        m = labels == j
        if m.any():
            new[j] = X[m].mean(0)
    return new

def inertia(X, labels, centres):
    return float(sum(((X[labels == j] - centres[j]) ** 2).sum() for j in range(len(centres))))

def isodata_ops(X, labels, centres, split_std, merge_dist, kmax, kmin):
    """One optional merge and one optional split per call. Returns (centres, events)."""
    events = []
    c = centres.copy()
    # ---- merge the closest pair, if too close ----
    if len(c) > kmin:
        dif = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((dif ** 2).sum(2))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        if dist[i, j] < merge_dist:
            mid = (c[i] + c[j]) / 2.0
            c = np.array([c[m] for m in range(len(c)) if m not in (i, j)] + [mid])
            events.append(("merge", f"clusters {i} and {j} were too close, merged into one"))
            return c, events
    # ---- split the most spread-out cluster, if too loose ----
    if len(c) < kmax:
        spreads = []
        for j in range(len(c)):
            pts = X[labels == j]
            spreads.append(pts.std(0).max() if len(pts) > 3 else 0.0)
        j = int(np.argmax(spreads))
        if spreads[j] > split_std:
            pts = X[labels == j]
            u, s, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
            axis = vt[0]; off = axis * spreads[j]
            c = np.array([c[m] for m in range(len(c)) if m != j] + [c[j] + off, c[j] - off])
            events.append(("split", f"cluster {j} was too spread out, split into two"))
    return c, events


# ============================================================================
# Streamlit app
# ============================================================================
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Week 9  The Clustering Loop", layout="wide")
    st.markdown(
        "<h2 style='margin-bottom:0'>The Clustering Loop, step by step</h2>"
        "<p style='color:#555;margin-top:4px'>The loop at the heart of ISODATA. "
        "Place the centres, then step through <b>assign</b> and <b>update</b> until nothing changes.</p>",
        unsafe_allow_html=True)

    # ---- sidebar controls ----
    sb = st.sidebar
    sb.header("Setup")
    dataset = sb.selectbox("Dataset", [
        "Coffs feature space (4 covers)", "Three clean blobs", "Overlapping / tricky"])
    n = sb.slider("Number of points", 60, 400, 160, step=20)
    k = sb.slider("Number of clusters (K)", 2, 10, 4,
                  help="Ask for more clusters than covers to see over-clustering, fewer to see under-clustering.")
    mode = sb.radio("Mode", ["K-means", "ISODATA (split and merge)"])
    if mode.startswith("ISODATA"):
        split_std = sb.slider("Split if spread above", 0.03, 0.20, 0.09, step=0.01)
        merge_dist = sb.slider("Merge if centres closer than", 0.03, 0.25, 0.10, step=0.01)
        kmax = sb.slider("Max clusters", k, 16, max(k + 4, 10))
    else:
        split_std, merge_dist, kmax = 1e9, 0.0, k
    colour_by = sb.radio("Colour points by", ["current cluster", "true cover"], horizontal=False)
    show_regions = sb.checkbox("Shade decision regions", value=True)
    seed = sb.number_input("Seed", 0, 9999, 7, step=1)
    if sb.button("New random start"):
        st.session_state.nonce = st.session_state.get("nonce", 0) + 1
    nonce = st.session_state.get("nonce", 0)

    # ---- (re)initialise the run when the setup changes ----
    sig = (dataset, n, k, int(seed), nonce, mode)
    if st.session_state.get("sig") != sig:
        X, truth, axes = make_points(dataset, n, int(seed) + nonce)
        centres = init_centres(X, k, int(seed) + nonce)
        st.session_state.update(
            sig=sig, X=X, truth=truth, axes=axes, centres=centres,
            labels=None, last_assign=None, it=0, nxt="assign",
            converged=False, events=[],
            log="Centres placed at random. Press **1. Assign** to start.")

    S = st.session_state
    X, truth, axes = S["X"], S["truth"], S["axes"]

    # ---- the step engine ----
    def step():
        if S["converged"]:
            return
        if S["nxt"] == "assign":
            labels = assign(X, S["centres"])
            if S["last_assign"] is not None and np.array_equal(labels, S["last_assign"]):
                S["converged"] = True
                S["log"] = "**Converged.** No point changed group, so no centre will move."
            else:
                changed = int((labels != S["last_assign"]).sum()) if S["last_assign"] is not None else len(X)
                S["labels"] = labels
                S["nxt"] = "update"
                S["events"] = []
                S["log"] = f"**Assign.** Every point took its nearest centre. {changed} points changed group."
        else:  # update
            new_c = update(X, S["labels"], S["centres"])
            note = "**Update.** Each centre moved to the middle of its points."
            S["events"] = []
            if mode.startswith("ISODATA"):
                new_c, ev = isodata_ops(X, S["labels"], new_c, split_std, merge_dist, kmax, 2)
                S["events"] = ev
                for kind, msg in ev:
                    note += f"  \n**{kind.upper()}:** {msg}."
            S["centres"] = new_c
            S["last_assign"] = S["labels"]
            S["it"] += 1
            S["nxt"] = "assign"
            S["log"] = note

    # ---- controls ----
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.2, 1])
    nxt_label = "1. Assign" if S["nxt"] == "assign" else "2. Update"
    if c1.button(f"Next step  ({nxt_label})", type="primary", disabled=S["converged"], use_container_width=True):
        step()
    if c2.button("Run to convergence", disabled=S["converged"], use_container_width=True):
        for _ in range(200):
            step()
            if S["converged"]:
                break
    if c3.button("Re-seed centres", use_container_width=True):
        S["centres"] = init_centres(X, k, int(seed) + nonce + S["it"] + 1)
        S.update(labels=None, last_assign=None, it=0, nxt="assign", converged=False,
                 events=[], log="Centres re-seeded. Press **1. Assign**.")
    if c4.button("Reset", use_container_width=True):
        st.session_state.pop("sig", None)
        st.rerun()

    # ---- metrics ----
    labels = S["labels"]
    inr = inertia(X, labels, S["centres"]) if labels is not None else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iteration", S["it"])
    m2.metric("Clusters", len(S["centres"]))
    m3.metric("Next move", "Assign" if S["nxt"] == "assign" else "Update")
    m4.metric("Inertia (spread)", "n/a" if labels is None else f"{inr:.3f}")

    # ---- the plot ----
    xlab, ylab, lim = axes
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor(PAPER); ax.set_facecolor("white")
    centres = S["centres"]
    if show_regions and len(centres):
        gx, gy = np.meshgrid(np.linspace(*lim, 300), np.linspace(*lim, 300))
        G = np.c_[gx.ravel(), gy.ravel()]
        reg = assign(G, centres).reshape(gx.shape)
        cmap = matplotlib.colors.ListedColormap([CLUSTER_COLS[j % len(CLUSTER_COLS)] for j in range(len(centres))])
        ax.imshow(reg, extent=[lim[0], lim[1], lim[0], lim[1]], origin="lower",
                  cmap=cmap, alpha=0.12, aspect="auto", zorder=0,
                  vmin=0, vmax=max(len(centres) - 1, 1))
    if colour_by == "true cover" and dataset.startswith("Coffs"):
        for kk in range(4):
            m = truth == kk
            ax.scatter(X[m, 0], X[m, 1], s=20, color=CLASS_COLS[kk], edgecolor="white",
                       linewidth=0.4, alpha=0.9, zorder=3, label=CLASS_NAMES[kk])
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    elif labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=20, color=GREY, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    else:
        for j in range(len(centres)):
            m = labels == j
            ax.scatter(X[m, 0], X[m, 1], s=20, color=CLUSTER_COLS[j % len(CLUSTER_COLS)],
                       edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3)
    for j in range(len(centres)):
        ax.scatter([centres[j, 0]], [centres[j, 1]], s=320, marker="X",
                   color=CLUSTER_COLS[j % len(CLUSTER_COLS)], edgecolor=INK, linewidth=1.7, zorder=6)
    for kind, _ in S["events"]:
        ax.text(0.5, 1.02, kind.upper(), transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=(FOREST if kind == "split" else BARE))
    title = ("Centres placed. Next: assign." if labels is None else
             ("Converged." if S["converged"] else
              ("Assigned. Next: update the centres." if S["nxt"] == "update"
               else "Centres moved. Next: re-assign.")))
    ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=8)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(xlab, color=INK); ax.set_ylabel(ylab, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#ddd")

    left, right = st.columns([1.5, 1])
    left.pyplot(fig)
    with right:
        st.markdown("### What just happened")
        st.info(S["log"])
        if S["converged"]:
            st.success("The clustering has settled. Every cluster is a spectral group. "
                       "Now the analyst names the groups, which is the labelling step.")
        st.markdown("### The loop")
        st.markdown(
            "1. **Place centres** at random.\n"
            "2. **Assign** every point to its nearest centre.\n"
            "3. **Update** each centre to the middle of its points.\n"
            "4. **Repeat** until nothing changes.")
        if mode.startswith("ISODATA"):
            st.markdown(
                "**ISODATA** adds two moves between rounds:\n"
                "- **Split** a cluster that is too spread out.\n"
                "- **Merge** two clusters that are too close.\n"
                "so the number of clusters adjusts itself.")
        st.caption("Try it: set K above or below the number of covers to see over and "
                   "under-clustering. In ISODATA mode, start K high (say 8 on the Coffs "
                   "data) and watch it merge down to the natural number, or start low and "
                   "watch it split up. Switch to 'true cover' colouring to compare the "
                   "clusters against what the covers really are.")

    plt.close(fig)


if __name__ == "__main__":
    run_app()
