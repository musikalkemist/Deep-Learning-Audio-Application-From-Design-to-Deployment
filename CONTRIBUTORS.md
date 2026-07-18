# Project Contributors

Thank you to everyone who has contributed to the success of this project! Your efforts, whether large or small, are greatly appreciated.

---

## Community & Project Maintainers

* **musikalkemist:** Author and project creator.
* **HimanshuKGP007:** Project reviewer.
* **MLsound:** Project maintainer.
    - **Refactoring & Maintenance:**
        - Refreshed project documentation including `README.md`, `CONTRIBUTING.md`, and `CONTRIBUTORS.md`.
        - Standardized model file formats (migrated from HDF5 `.h5` to native Keras `.keras`).
        - Upgraded and pinned dependencies for Python 3.11 compatibility.
        - Unified folder naming and refined `.gitignore` rules.
    - **Version Management:** Established and organized the `legacy` branch to preserve the original course environment for students, maintaining compatibility with video content while bringing the main branch to modern standards.
    - **Features & Automation:**
        - Implemented an automated dataset downloader (`utils/dataset_downloader.py`) and setup script (`setup_dataset.sh`) for the Google Speech Commands dataset.
        - Integrated automated uWSGI/Docker configurations and Docker/NGINX build stability.
        - Developed AWS deployment automation scripts (`deploy.sh`, `init.sh`), guides, and troubleshooting resources.
        - Built file propagation (`utils/propagate_files.py`) and file removal utilities for multi-folder codebase sync.

### Contributors

* **geopapa11:** Added PyTorch model compatibility and training pipeline (PR #31).
    - Implemented PyTorch model training module (`train_pt.py`) and keyword spotting service (`keyword_spotting_service_pt.py`).
    - Enhanced Flask server (`server.py`) to support both TensorFlow and PyTorch models for inference.
    - Added command-line support to `client.py` for flexible audio filepath and model type selections.
    - Simplified dependencies, Dockerfile configs, and polished preprocessing scripts.

---

Want to see your name on this list? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file to learn how you can help!
