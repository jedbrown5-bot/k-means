# Week 9 game: The Clustering Loop, step by step

An interactive Streamlit visualiser for the loop at the heart of ISODATA, built for
the SPA317 / SPA442 Week 9 lecture (Unsupervised and Hybrid Classification).

Students place the centres, then step through the two moves one at a time:

1. **Assign** every point to its nearest centre (nearest centre wins).
2. **Update** each centre to the middle of its points.

and repeat until nothing changes (convergence). An **ISODATA** mode adds the split
and merge moves, so the number of clusters adjusts itself.

## Run it

You have Anaconda already, so from a terminal (or the Anaconda Prompt):

```
pip install streamlit streamlit-image-coordinates
streamlit run Week9_kmeans_game.py
```

It opens in your browser. Nothing is uploaded; it runs entirely on your machine. The
`streamlit-image-coordinates` package is optional: it enables the "Click to place" mode.
Without it, the app still runs and "Type coordinates" placement still works.

## Placing the centres

Use the **Centre start** control in the sidebar:

- **Random** — the machine drops the centres (k-means++ seeding).
- **Click to place** — click anywhere in the feature-space box to drop each centre. Place at least two, then step.
- **Type coordinates** — type or edit centre coordinates in a small table (no extra package needed).

Placing the centres badly on purpose, then watching the loop still find the groups, is a good way to show that the result is robust to where the centres start.

## What to try in class

- Step through one move at a time with **Next step**, reading the "What just happened" panel each time.
- Set **K** above or below the number of covers to show over-clustering and under-clustering (ties to the "how many clusters" slide).
- Switch **Dataset** to "Overlapping / tricky" to show where plain K-means struggles.
- Turn on **ISODATA (split and merge)**, start with K set high (say 8 on the Coffs data), and watch the cluster count merge its way down to the natural number.
- Switch **Colour points by** to "true cover" (Coffs dataset) to compare the machine's clusters against the real covers, which is the labelling step.
- Turn on **Number the clusters** to badge each centre with its number, matching the Centres table on the right, so you can point to "cluster 3" and everyone knows which one you mean.
- Turn on **Show cursor coordinates** to read the feature-space position under the cursor as you move over the plot. The reading is the position on the two axes, so you can show where a water pixel or a bright bare-soil pixel actually sits.

## Files

- `Week9_kmeans_game.py` — the app (single file; the clustering logic is pure functions at the top, the Streamlit UI below).
- `requirements.txt` — dependencies.
