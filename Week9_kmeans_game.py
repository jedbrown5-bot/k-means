"""
SPA317 / SPA442  Week 9  The Clustering Loop, step by step.

An interactive visualiser for the loop at the heart of ISODATA. Students place
the centres (at random, by clicking, or by typing coordinates), then step through
the two moves one at a time:
  1. Assign  every point to its nearest centre  (nearest centre wins)
  2. Update  each centre to the middle of its points
and repeat until nothing changes (convergence). An ISODATA mode adds the split
and merge moves, so the number of clusters adjusts itself.

Run it with:   streamlit run Week9_kmeans_game.py

Clicking to place centres needs one extra package:
   pip install streamlit-image-coordinates
Without it, "Click to place" is hidden but "Type coordinates" still works.
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
CLUSTER_COLS = [WATER, FOREST, BARE, "#7b3fa0", PASTURE, "#c0392b", "#00838f",
                "#8d6e63", "#d81b60", "#3949ab", "#f9a825", "#546e7a",
                "#26a69a", "#ad1457", "#5d4037", "#1565c0"]

# ============================================================================
# Pure clustering logic (no Streamlit here, so it can be tested on its own)
# ============================================================================
def make_points(dataset, n, seed):
    """Return (X, truth, axes): n points in 2D, their true-cover labels, and axis info."""
    rng = np.random.default_rng(seed)
    if dataset == "Coffs feature space (4 covers)":
        specs = [((0.06, 0.09), (0.020, 0.020)),
                 ((0.12, 0.42), (0.030, 0.055)),
                 ((0.30, 0.34), (0.045, 0.040)),
                 ((0.40, 0.16), (0.038, 0.032))]
        xlab, ylab, lim = "Red", "NIR", (0.0, 0.50)
    elif dataset == "Three clean blobs":
        specs = [((0.22, 0.24), (0.045, 0.045)),
                 ((0.32, 0.72), (0.050, 0.050)),
                 ((0.72, 0.40), (0.055, 0.055))]
        xlab, ylab, lim = "feature 1", "feature 2", (0.0, 0.95)
    else:  # Overlapping / tricky
        specs = [((0.35, 0.45), (0.11, 0.030)),
                 ((0.45, 0.62), (0.055, 0.055)),
                 ((0.55, 0.60), (0.055, 0.055)),
                 ((0.72, 0.28), (0.050, 0.050))]
        xlab, ylab, lim = "feature 1", "feature 2", (0.0, 1.0)
    per = max(1, n // len(specs))
    X, truth = [], []
    for k, ((cx, cy), (sx, sy)) in enumerate(specs):
        X.append(np.column_stack([rng.normal(cx, sx, per), rng.normal(cy, sy, per)]))
        truth += [k] * per
    X = np.clip(np.vstack(X), lim[0], lim[1])
    return X, np.array(truth), (xlab, ylab, lim)

def init_centres(X, k, seed):
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

def pixel_to_data(px, py, w, h, lim):
    """Map a click at image pixel (px, py) to data coordinates for a full-bleed square axes."""
    lo, hi = lim
    x = lo + (px / w) * (hi - lo)
    y = hi - (py / h) * (hi - lo)   # image y runs downward, data y upward
    return float(np.clip(x, lo, hi)), float(np.clip(y, lo, hi))

def isodata_ops(X, labels, centres, split_std, merge_dist, kmax, kmin):
    events = []
    c = centres.copy()
    if len(c) > kmin:
        dif = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((dif ** 2).sum(2))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        if dist[i, j] < merge_dist:
            mid = (c[i] + c[j]) / 2.0
            c = np.array([c[m] for m in range(len(c)) if m not in (i, j)] + [mid])
            events.append(("merge", f"two centres were too close, merged into one"))
            return c, events
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
            events.append(("split", f"one cluster was too spread out, split into two"))
    return c, events


# ============================================================================
# Streamlit app
# ============================================================================
def run_app():
    import streamlit as st
    from io import BytesIO
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        HAVE_CLICK = True
    except Exception:
        HAVE_CLICK = False

    st.set_page_config(page_title="Week 9  The Clustering Loop", layout="wide")
    st.markdown(
        "<h2 style='margin-bottom:0'>The Clustering Loop, step by step</h2>"
        "<p style='color:#555;margin-top:4px'>The loop at the heart of ISODATA. "
        "Place the centres, then step through <b>assign</b> and <b>update</b> until nothing changes.</p>",
        unsafe_allow_html=True)

    S = st.session_state

    def run_started():
        return S.get("it", 0) > 0 or S.get("labels") is not None

    def reset_run(centres):
        S["centres"] = np.asarray(centres, dtype=float).reshape(-1, 2)
        S.update(labels=None, last_assign=None, it=0, nxt="assign",
                 converged=False, events=[],
                 log="Centres ready. Press **1. Assign** to start.")

    # ---- sidebar ----
    sb = st.sidebar
    sb.header("Setup")
    dataset = sb.selectbox("Dataset", [
        "Coffs feature space (4 covers)", "Three clean blobs", "Overlapping / tricky"])
    n = sb.slider("Number of points", 60, 400, 160, step=20)
    starts = ["Random"] + (["Click to place"] if HAVE_CLICK else []) + ["Type coordinates"]
    centre_start = sb.radio("Centre start", starts,
                            help="Let the machine drop the centres, or place them yourself.")
    manual = centre_start != "Random"
    if not manual:
        k = sb.slider("Number of clusters (K)", 2, 10, 4,
                      help="Ask for more clusters than covers to see over-clustering, fewer for under.")
    else:
        k = None
        sb.caption("You choose the centres. K is however many you place. Place at least 2.")
        if centre_start == "Click to place" and not HAVE_CLICK:
            sb.warning("Run 'pip install streamlit-image-coordinates' to enable clicking.")
    mode = sb.radio("Mode", ["K-means", "ISODATA (split and merge)"])
    if mode.startswith("ISODATA"):
        split_std = sb.slider("Split if spread above", 0.03, 0.20, 0.09, step=0.01)
        merge_dist = sb.slider("Merge if centres closer than", 0.03, 0.25, 0.10, step=0.01)
        kmax = sb.slider("Max clusters", 2, 16, 12)
    else:
        split_std, merge_dist, kmax = 1e9, 0.0, 99
    colour_by = sb.radio("Colour points by", ["current cluster", "true cover"])
    show_regions = sb.checkbox("Shade decision regions", value=True)
    seed = sb.number_input("Seed", 0, 9999, 7, step=1)
    if sb.button("New random layout"):
        S["nonce"] = S.get("nonce", 0) + 1
        S.pop("pkey", None)
    nonce = S.get("nonce", 0)

    # ---- regenerate points when the layout changes ----
    pkey = (dataset, n, int(seed), nonce)
    if S.get("pkey") != pkey:
        X, truth, axes = make_points(dataset, n, int(seed) + nonce)
        S.update(pkey=pkey, X=X, truth=truth, axes=axes, placed=[], last_click=None)
        S.pop("cfg", None)
    X, truth, axes = S["X"], S["truth"], S["axes"]
    xlab, ylab, lim = axes

    # ---- decide the initial centres ----
    if not manual:
        cfg = (pkey, "rand", k, int(S.get("reseed", 0)))
        if S.get("cfg") != cfg:
            S["cfg"] = cfg
            reset_run(init_centres(X, k, int(seed) + nonce + int(S.get("reseed", 0))))
    else:
        # manual: while the run has not started, keep centres synced to placement
        placed = S.get("placed", [])
        if not run_started():
            cfg = (pkey, centre_start, tuple(map(tuple, placed)))
            if S.get("cfg") != cfg:
                S["cfg"] = cfg
                reset_run(np.array(placed, dtype=float) if placed else np.zeros((0, 2)))

    centres = S["centres"]
    can_step = len(centres) >= 2

    # ---- the step engine ----
    def step():
        if S["converged"] or len(S["centres"]) < 2:
            return
        if S["nxt"] == "assign":
            labels = assign(X, S["centres"])
            if S["last_assign"] is not None and np.array_equal(labels, S["last_assign"]):
                S["converged"] = True
                S["log"] = "**Converged.** No point changed group, so no centre will move."
            else:
                changed = int((labels != S["last_assign"]).sum()) if S["last_assign"] is not None else len(X)
                S.update(labels=labels, nxt="update", events=[],
                         log=f"**Assign.** Every point took its nearest centre. {changed} points changed group.")
        else:
            new_c = update(X, S["labels"], S["centres"])
            note = "**Update.** Each centre moved to the middle of its points."
            S["events"] = []
            if mode.startswith("ISODATA"):
                new_c, ev = isodata_ops(X, S["labels"], new_c, split_std, merge_dist, kmax, 2)
                S["events"] = ev
                for kind, msg in ev:
                    note += f"  \n**{kind.upper()}:** {msg}."
            S.update(centres=new_c, last_assign=S["labels"], it=S["it"] + 1, nxt="assign", log=note)

    # ---- controls ----
    # NB: keep button LABELS constant (with stable keys). A changing label makes
    # Streamlit treat it as a new widget and drop every other click.
    c1, c2, c3, c4 = st.columns([1.4, 1.3, 1.2, 1])
    if c1.button("Next step", key="btn_next", type="primary",
                 disabled=S["converged"] or not can_step, use_container_width=True):
        step()
    if c2.button("Run to convergence", key="btn_run",
                 disabled=S["converged"] or not can_step, use_container_width=True):
        for _ in range(200):
            step()
            if S["converged"]:
                break
    if c3.button("Re-seed centres", key="btn_reseed", disabled=manual, use_container_width=True,
                 help="Random mode only. In manual mode use Clear and place again."):
        S["reseed"] = int(S.get("reseed", 0)) + 1
        S.pop("cfg", None)
    if c4.button("Reset", key="btn_reset", use_container_width=True):
        for key in ["pkey", "cfg"]:
            S.pop(key, None)
        S["placed"] = []
        st.rerun()

    if S["converged"]:
        st.caption("Converged. Press Reset to start again.")
    elif manual and not can_step:
        st.info("Place at least 2 centres to start stepping.")
    else:
        nxt_word = "Assign (points take their nearest centre)" if S["nxt"] == "assign" \
            else "Update (centres move to the middle of their points)"
        st.caption(f"Next step will: **{nxt_word}**.")

    # ---- metrics ----
    labels = S["labels"]
    inr = inertia(X, labels, S["centres"]) if labels is not None and len(S["centres"]) else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iteration", S["it"])
    m2.metric("Clusters", len(S["centres"]))
    m3.metric("Next move", "Assign" if S["nxt"] == "assign" else "Update")
    m4.metric("Inertia (spread)", "n/a" if labels is None else f"{inr:.3f}")

    left, right = st.columns([1.5, 1])

    # ---- placement pad (click mode, during setup) OR the styled plot ----
    placing_by_click = manual and centre_start == "Click to place" and HAVE_CLICK and not run_started()

    if placing_by_click:
        W = 460
        figp = plt.figure(figsize=(W / 100, W / 100), dpi=100)
        axp = figp.add_axes([0, 0, 1, 1]); axp.set_xlim(*lim); axp.set_ylim(*lim); axp.axis("off")
        axp.set_facecolor("white")
        if show_regions and len(centres):
            gx, gy = np.meshgrid(np.linspace(*lim, 250), np.linspace(*lim, 250))
            reg = assign(np.c_[gx.ravel(), gy.ravel()], centres).reshape(gx.shape)
            cmap = matplotlib.colors.ListedColormap([CLUSTER_COLS[j % len(CLUSTER_COLS)] for j in range(len(centres))])
            axp.imshow(reg, extent=[lim[0], lim[1], lim[0], lim[1]], origin="lower",
                       cmap=cmap, alpha=0.12, aspect="auto", vmin=0, vmax=max(len(centres) - 1, 1))
        axp.scatter(X[:, 0], X[:, 1], s=18, color=GREY, edgecolor="white", linewidth=0.4, alpha=0.85)
        for i, (cx, cy) in enumerate(centres):
            axp.scatter([cx], [cy], s=300, marker="X", color=CLUSTER_COLS[i % len(CLUSTER_COLS)],
                        edgecolor=INK, linewidth=1.7)
        buf = BytesIO(); figp.savefig(buf, format="png", facecolor="white"); plt.close(figp)
        from PIL import Image
        buf.seek(0); pad_img = Image.open(buf)
        with left:
            st.markdown(f"**Click in the box to drop a centre.**  ({len(centres)} placed)")
            coords = streamlit_image_coordinates(pad_img, width=W, key="pad")
            st.caption(f"Axes: {xlab} across, {ylab} up. Place at least 2, then press Next step.")
            if st.button("Clear centres"):
                S["placed"] = []; S["last_click"] = None; S.pop("cfg", None); st.rerun()
        if coords is not None and coords != S.get("last_click"):
            S["last_click"] = coords
            dx, dy = pixel_to_data(coords["x"], coords["y"], W, W, lim)
            S["placed"] = S.get("placed", []) + [[dx, dy]]
            S.pop("cfg", None)
            st.rerun()
    else:
        # ---- the styled plot ----
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        fig.patch.set_facecolor(PAPER); ax.set_facecolor("white")
        if show_regions and len(centres):
            gx, gy = np.meshgrid(np.linspace(*lim, 300), np.linspace(*lim, 300))
            reg = assign(np.c_[gx.ravel(), gy.ravel()], centres).reshape(gx.shape)
            cmap = matplotlib.colors.ListedColormap([CLUSTER_COLS[j % len(CLUSTER_COLS)] for j in range(len(centres))])
            ax.imshow(reg, extent=[lim[0], lim[1], lim[0], lim[1]], origin="lower",
                      cmap=cmap, alpha=0.12, aspect="auto", zorder=0, vmin=0, vmax=max(len(centres) - 1, 1))
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
        if len(centres) < 2:
            title = "Place at least 2 centres."
        elif labels is None:
            title = "Centres placed. Next: assign."
        elif S["converged"]:
            title = "Converged."
        elif S["nxt"] == "update":
            title = "Assigned. Next: update the centres."
        else:
            title = "Centres moved. Next: re-assign."
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=8)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(xlab, color=INK); ax.set_ylabel(ylab, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#ddd")
        left.pyplot(fig)
        plt.close(fig)

    # ---- type-coordinates editor (manual, during setup) ----
    if manual and centre_start == "Type coordinates" and not run_started():
        import pandas as pd
        with right:
            st.markdown("### Type the centres")
            st.caption(f"One row per centre. {xlab} (x) and {ylab} (y) in the range "
                       f"{lim[0]:.2f} to {lim[1]:.2f}. Add or delete rows to change how many.")
            default = S.get("placed") or [[lim[1] * 0.3, lim[1] * 0.3], [lim[1] * 0.6, lim[1] * 0.6]]
            df = pd.DataFrame(default, columns=["x", "y"])
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="coord_editor")
            new_placed = [[float(np.clip(r.x, *lim)), float(np.clip(r.y, *lim))]
                          for r in edited.itertuples() if pd.notna(r.x) and pd.notna(r.y)]
            if new_placed != S.get("placed"):
                S["placed"] = new_placed; S.pop("cfg", None); st.rerun()

    # ---- explanation panel ----
    with right:
        st.markdown("### What just happened")
        st.info(S["log"])
        if S["converged"]:
            st.success("The clustering has settled. Every cluster is a spectral group. "
                       "Now the analyst names the groups, which is the labelling step.")
        st.markdown("### The loop")
        st.markdown(
            "1. **Place centres** (random, click, or type).\n"
            "2. **Assign** every point to its nearest centre.\n"
            "3. **Update** each centre to the middle of its points.\n"
            "4. **Repeat** until nothing changes.")
        if mode.startswith("ISODATA"):
            st.markdown(
                "**ISODATA** adds two moves between rounds:\n"
                "- **Split** a cluster that is too spread out.\n"
                "- **Merge** two clusters that are too close.")
        st.caption("Try it: place the centres badly on purpose and watch the loop still find the groups, "
                   "set K above or below the number of covers, or turn on ISODATA and watch the count settle.")


if __name__ == "__main__":
    run_app()
