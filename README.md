# AI for Chemistry Coursework

The brief for the coursework to be provided to students is in the [COURSEWORK.md](COURSEWORK.md) document.

Students can run everything on Google Colab or locally.

## Quick start (Colab)

1. Click one of the badges above to open a notebook in Colab.
2. In the first cell of **either** notebook, run:

```bash
!git clone https://github.com/mdi-group/ai-for-chemistry-coursework-students.git
%cd ai-for-chemistry-coursework-students
!bash scripts/setup_colab.sh
```

3. If this is the first time doing the coursework, then download your unique dataset:

```
!python scripts/download-setup-unique-dataset.py
```
> The provided data script generates a **unique, randomly mutated dataset per run** and currently writes the output file to the repo root (e.g. `student_dataset_ab12cd34.pkl`).

4. Proceed with the notebook.

5. **Important note** if you are restarting the coursework you will have to save the unique dataset pkl file that you generated and load it back up - i.e. you do not need to run the `!python scripts/download-setup-unique-dataset.py` command. You will have to clone the repo and setup the colab environment each time though.

## Building features

There is a helper function to assist with converting the dataframe that you have into some kind of features. To use it follow this code:

```
from src.utils import add_composition_column
from matminer.featurizers.composition import ElementProperty

df = add_composition_column(df)
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df_with_features = ep_feat.featurize_dataframe(df, col_id="composition")
```

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Project structure

- `scripts/download-setup-unique-dataset.py` — custom script that downloads and mutates the dataset.
- `scripts/setup_colab.sh` — installs Python deps in Colab runtime.
- `src/utils.py`, `src/utils_gnn.py` — helper utilities used by the notebooks.

## Python version

Tested on Python 3.10+ (Colab’s default is fine).

## License

MIT (see `LICENSE`).
