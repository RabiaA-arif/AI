# AI and Machine Learning Workbench

A practical collection of artificial intelligence, machine-learning, natural-language-processing, API, and deployment experiments. Each major folder contains an individual project, notebook, application, or prototype.

> **Project status:** This repository is primarily a learning and experimentation workspace. Some projects may be incomplete or use different dependencies. Read the project-level documentation before running any individual project.

## Repository contents

| Area | Path | Description |
|---|---|---|
| Fake-news detection | [`fake_news_detector/`](./fake_news_detector) | Experiments and code for detecting potentially misleading or fake news. |
| Credit-risk prediction | [`credit_risk_prediction/`](./credit_risk_prediction) | Machine-learning work for estimating credit risk. See the [project README](./credit_risk_prediction/README.md). |
| Machine-learning algorithms | [`machine_learning_algorithms/`](./machine_learning_algorithms) | Practical implementations and experiments with machine-learning algorithms. |
| Air-quality prediction | [`air_quality_index_prediction/`](./air_quality_index_prediction) | Data collection, cleaning, and analysis related to air-quality-index prediction. |
| Medical chatbot | [`medical_chatbot/`](./medical_chatbot) | Chatbot application code and supporting research materials. |
| Natural-language processing | [`nlp/`](./nlp) | NLP practice and experimental work. |
| Docker wine predictor | [`Docker/wine-predictor/`](./Docker/wine-predictor) | A machine-learning application packaged for Docker deployment. |
| FastAPI projects | [`FastAPI/`](./FastAPI) | API practice and machine-learning model-serving projects. |
| Financial application | [`fmp-financial-app`](./fmp-financial-app) | A linked project included in this workspace. |
| Notebook experiments | [`assignment.ipynb`](./assignment.ipynb), [`colab_point.ipynb`](./colab_point.ipynb), [`nlp_paper_practice.ipynb`](./nlp_paper_practice.ipynb), [`nlp_practise.ipynb`](./nlp_practise.ipynb) | Exploratory notebooks and practice exercises. |

## Featured projects

### House-price prediction API

The [`FastAPI/house_price_api/`](./FastAPI/house_price_api) project trains a `RandomForestRegressor` on the scikit-learn California Housing dataset and exposes predictions through FastAPI. Its project-level documentation describes training, evaluation, single predictions, and CSV-based batch predictions.

Read [`FastAPI/house_price_api/readme.md`](./FastAPI/house_price_api/readme.md) before running it. The typical development commands are:

```bash
cd FastAPI/house_price_api
python train.py
uvicorn main:app --reload
```

When the API is running, its interactive documentation is normally available at [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

### Docker wine predictor

The [`Docker/wine-predictor/`](./Docker/wine-predictor) directory includes an application, a `Dockerfile`, and a `requirements.txt` file.

```bash
cd Docker/wine-predictor
docker build -t wine-predictor .
docker run --rm -p 8000:8000 wine-predictor
```

If the application listens on another port, update the `-p` mapping to match the configuration in [`app.py`](./Docker/wine-predictor/app.py).

### FastAPI projects

The [`FastAPI/`](./FastAPI) directory includes several API projects:

- [`fastapi_project/`](./FastAPI/fastapi_project)
- [`house_price_api/`](./FastAPI/house_price_api)
- [`practice_project/`](./FastAPI/practice_project)
- [`practise/`](./FastAPI/practise)
- [`production-ready-project/`](./FastAPI/production-ready-project)

Run each project from its own directory because the entry point and dependencies may differ.

## Getting started

### Prerequisites

Install the following tools as needed:

- Python 3.10 or newer, unless a project specifies another version.
- `pip` and `venv` for Python dependency management.
- Git for cloning and version control.
- Docker Desktop or Docker Engine for the Docker project.
- Jupyter Notebook or Google Colab for notebook-based experiments.

### Clone the repository

```bash
git clone https://github.com/RabiaA-arif/AI.git
cd AI
```

### Create a virtual environment

Use a separate environment for projects with different dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

Install dependencies from a project-specific file when one exists:

```bash
pip install -r path/to/requirements.txt
```

For example:

```bash
pip install -r Docker/wine-predictor/requirements.txt
```

### Run a notebook

```bash
jupyter notebook
```

Then select the notebook you want to explore. Some notebooks may require their original dataset paths or uploaded files.

## Recommended workflow

1. Choose a project folder and read its local README first.
2. Create or activate an environment for that project.
3. Install the project’s dependencies.
4. Check dataset paths, model files, environment variables, and required services.
5. Run preprocessing or training before starting an API, when applicable.
6. Test the application with its documented example input.
7. Record reproducible commands and results in the project documentation.

## Contributing

Documentation, organization, reproducibility, and code improvements are welcome. To propose a change, create a branch instead of working directly on `main`:

```bash
git checkout -b docs/describe-your-change
```

After making and testing your changes, commit and push the branch:

```bash
git add README.md
git commit -m "Improve repository documentation"
git push -u origin docs/describe-your-change
```

Then open a pull request against the `main` branch. Link the relevant issue with `Closes #ISSUE_NUMBER` when appropriate.

## License

No root-level license is currently documented. If you intend others to reuse, modify, or distribute this work, add an appropriate `LICENSE` file and update this section.

## Author

Maintained by [Rabia Arif](https://github.com/RabiaA-arif).

## Useful links

- [Repository](https://github.com/RabiaA-arif/AI)
- [Issues](https://github.com/RabiaA-arif/AI/issues)
- [Pull requests](https://github.com/RabiaA-arif/AI/pulls)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Docker documentation](https://docs.docker.com/)

## References

[1]: https://github.com/RabiaA-arif/AI "RabiaA-arif/AI repository"
[2]: https://fastapi.tiangolo.com/ "FastAPI documentation"
[3]: https://scikit-learn.org/stable/ "scikit-learn documentation"
[4]: https://docs.docker.com/ "Docker documentation"

