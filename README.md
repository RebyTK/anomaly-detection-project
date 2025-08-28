# Samsung EnnovateX 2025 AI Challenge Submission

Problem Statement - On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection
Team name - Rishabh and Raghav
Team members (Names) - Raghav Bagai, Rishabh Garg
Demo Video Link - 

## 🚀 Features

- Behavior model training with rich datasets
- Real-time anomaly inference
- Batch processing capabilities
- On-device behavior analysis

## 📁 Project Structure

```anomaly-detection-project/
├── generate_rich_behavior_dataset.py    # Dataset generation
├── train_behavior_model.py              # Basic behavior model training
├── train_behavior_rich_model.py         # Rich behavior model training
├── inference_rich_behavior.py           # Real-time inference
├── infer_rich_behavior_batch.py         # Batch inference
└── ondevice-behaviour/                  # On-device behavior analysis
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/anomaly-detection-project.git
cd anomaly-detection-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training Models
```bash
python train_behavior_model.py
python train_behavior_rich_model.py
```

### Generating Datasets
```bash
python generate_rich_behavior_dataset.py
```

### Running Inference
```bash
python inference_rich_behavior.py
python infer_rich_behavior_batch.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Project Link: [https://github.com/RebyTK/anomaly-detection-project](https://github.com/RebyTK/anomaly-detection-project)

## 4. Create a .gitignore File

Make sure you have a proper `.gitignore` file for Python projects:

```gitignore:.gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

## 5. Create a Requirements.txt

If you don't have one already, create a `requirements.txt` file with your project dependencies:

```txt:requirements.txt
# Add your project dependencies here
# For example:
# numpy>=1.21.0
# pandas>=1.3.0
# scikit-learn>=1.0.0
# torch>=1.9.0
# matplotlib>=3.4.0
# seaborn>=0.11.0
```

## 6. Additional Files to Consider

- **LICENSE**: Choose an appropriate license (MIT, Apache, GPL, etc.)
- **CONTRIBUTING.md**: Guidelines for contributors
- **CHANGELOG.md**: Track version changes
- **docs/**: Documentation folder
- **tests/**: Unit tests folder

## 7. Push Your Changes

After creating these files:

```bash
git add .
git commit -m "Add project documentation and setup files"
git push origin main
```

## 8. Set Up GitHub Features

- **Issues**: Enable issue tracking for bugs and feature requests
- **Projects**: Create project boards for task management
- **Actions**: Set up CI/CD workflows if needed
- **Wiki**: Add project documentation if desired

