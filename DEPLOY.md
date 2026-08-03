# Hosting the Week 9 game as a live Streamlit app

The app is already a Streamlit app. This note is about putting it on the web so
students open a link instead of installing anything. It is fully self-contained
(no data files, no logins), so it deploys in a few minutes on a free host.

Files the host needs (all in this folder):

- `streamlit_app.py`  (the entry point)
- `Week9_kmeans_game.py`  (the app)
- `requirements.txt`  (dependencies, including the click component)
- `.streamlit/config.toml`  (the house theme; optional but nice)

## Option A: Streamlit Community Cloud (recommended, free)

This is Streamlit's own free hosting. It needs the files in a public GitHub repo.

1. Create a GitHub account if you do not have one, at github.com.
2. Make a new public repository, for example `spa317-clustering-game`. On the new
   repo page choose "uploading an existing file", then drag in `streamlit_app.py`,
   `Week9_kmeans_game.py`, and `requirements.txt`. Create the `.streamlit` folder by
   typing `.streamlit/config.toml` as the filename when you add that file, and paste
   its contents. Commit.
3. Go to share.streamlit.io and sign in with GitHub.
4. Click "New app", pick your repository and branch, set the main file path to
   `streamlit_app.py`, and click "Deploy".
5. It installs the requirements, builds, and gives you a public URL like
   `https://spa317-clustering-game.streamlit.app` that you can put in Brightspace.

The click-to-place feature works on Community Cloud because
`streamlit-image-coordinates` is listed in `requirements.txt` and installs automatically.

To update the app later, edit the file in GitHub (or push a change) and Community
Cloud redeploys on its own.

## Option B: Hugging Face Spaces (free, no git needed)

If you would rather not touch GitHub, Hugging Face lets you upload files in the browser.

1. Create a free account at huggingface.co.
2. Click your avatar, then "New Space". Give it a name, choose SDK "Streamlit",
   set it Public, and create it.
3. In the Space, open the "Files" tab and upload `streamlit_app.py`,
   `Week9_kmeans_game.py`, `requirements.txt`, and `.streamlit/config.toml`
   (use "Add file", then "Upload files").
4. Open the auto-created `README.md` in the Space and make sure the front matter at
   the top has `app_file: streamlit_app.py` and `sdk: streamlit`. Save.
5. The Space builds and serves at `https://huggingface.co/spaces/<your-name>/<space>`.

## Option C: A quick temporary link for one class (no deploy)

To share your locally running app for a single session without deploying:

1. Run it as usual: `streamlit run streamlit_app.py`.
2. In another terminal, expose port 8501 with a tunnel, for example
   `npx localtunnel --port 8501` (or ngrok if you have it), which prints a temporary
   public URL. It stays live only while your machine runs the app.

## Notes

- Python version: the hosts default to a recent Python 3, which is fine. No pinning needed.
- If Community Cloud complains it cannot find the app, check the main file path is
  exactly `streamlit_app.py`.
- Nothing in the app writes files or needs secrets, so it is safe to host publicly.
