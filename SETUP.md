# Mixture Studio — Streamlit Setup

## Installation

1. **Clone or download the files:**
   - `mixture_doe.py` (the numerics core)
   - `streamlit_app.py` (the Streamlit interface)
   - `mixture_studio.html` (optional, for reference — not needed for Streamlit)

2. **Install dependencies:**
   ```bash
   pip install streamlit numpy matplotlib mpltern pandas
   ```

   If `mpltern` fails to install (it's a lesser-known package), you can get by without it temporarily:
   ```bash
   pip install streamlit numpy matplotlib pandas
   # mpltern will be needed for ternary plots; if it fails, see troubleshooting below
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

   Streamlit will open a browser window to `http://localhost:8501` with the app live.

## Workflow

1. **Set components & constraints** (left panel):
   - Name your three components (Surfactant, Oil, Co-Surfactant, etc.)
   - Enter min/max bounds for each

2. **Suggest design points** (Step 2):
   - Pick a budget (4–8 points)
   - Hit "Suggest points" to get the D-optimal design
   - The ternary plot shows your feasible region and design points

3. **Enter solubility data** (Step 3):
   - Measure the response for each run in your experiment
   - Type values into the solubility inputs

4. **Fit model** (Step 4):
   - Click "Fit model"
   - Diagnostics appear: R², adjusted R², LOO-RMSE, and an honest verdict (under-determined / poor fit / trustworthy)
   - The ternary plot draws a viridis heatmap of predicted solubility

5. **Export/import data**:
   - Download your results as CSV
   - Upload a previous session's CSV to resume work

## Troubleshooting

### `mpltern` won't install
This is a known issue on some systems. Try:
```bash
pip install --upgrade pip
pip install mpltern --no-cache-dir
```

If it still fails, you can comment out the `import mpltern` line in `streamlit_app.py` and the plot will still display a basic feasible-region outline without the heatmap surface. The diagnostics and design selection still work fully.

### "streamlit not found"
Make sure you installed to the right Python:
```bash
which python    # Check which Python you're using
python -m pip install streamlit  # Install via the module interface to be sure
```

### Port 8501 is in use
Streamlit defaults to port 8501. To use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### App is slow
This is expected on first load (imports mpltern, fits models). Once running, interactions are snappy. If you see "rerunning" constantly, check that you're not accidentally editing the file — Streamlit auto-reruns on save.

## File structure

```
.
├── mixture_doe.py           # Core numerics (vertices, D-optimal, Scheffé model)
├── streamlit_app.py         # Streamlit UI (controls, plots, state)
├── mixture_studio.html      # Standalone HTML (reference, optional)
└── SETUP.md                 # This file
```

## Deployment

To share this app online:

**Streamlit Community Cloud (free & easiest):**
1. Push your files to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select the repo
4. Streamlit builds & hosts for free

**Self-hosted (Docker, server, etc.):**
```bash
# Create requirements.txt
pip freeze > requirements.txt

# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]

# Build & run
docker build -t mixture-studio .
docker run -p 8501:8501 mixture-studio
```

## Questions?

- **Streamlit docs:** https://docs.streamlit.io
- **mixture_doe module:** See docstrings in `mixture_doe.py`
- **Ternary plots:** https://mpltern.readthedocs.io (or just use `matplotlib.pyplot` if mpltern is giving trouble)

Enjoy!
