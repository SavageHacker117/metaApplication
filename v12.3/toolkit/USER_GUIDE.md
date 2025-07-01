
# V5 Toolkit User Guide

This guide provides detailed instructions on how to use the V5 Toolkit, including its core functionalities and advanced features. The toolkit is designed to streamline your development and testing workflows for the RL Tower Defense Code Synthesis project.

## 1. Toolkit Contents

The V5 Toolkit `.zip` file contains the following components:

- `demo_run.sh`: A bash script to run a V5 demo session and save all artifacts.
- `make_gif.py`: A Python script to automatically generate GIFs from agent output images.
- `stress_test.sh`: A bash script to simulate various stress scenarios on the V5 system.
- `config_manager.sh`: A bash script for managing V5 training configurations.
- `README.md`: The main README file providing an overview of the V5 project and toolkit.
- `USER_GUIDE.md`: This comprehensive user guide.

## 2. Getting Started

To begin using the toolkit, first ensure you have the necessary dependencies installed for your main project. The toolkit scripts assume a working Python environment with `Pillow` (for `make_gif.py`) and `torch` (for `stress_test.sh` CUDA OOM test) if you intend to use all features.

### 2.1. Executable Permissions

Ensure that the bash scripts (`demo_run.sh`, `stress_test.sh`, `config_manager.sh`) have executable permissions. If not, you can grant them using the `chmod` command:

```bash
chmod +x demo_run.sh
chmod +x stress_test.sh
chmod +x config_manager.sh
```

## 3. Core Toolkit Functionalities

### 3.1. Running a Demo Session (`demo_run.sh`)

The `demo_run.sh` script allows you to easily run a V5 demo session. It automatically creates a unique output directory for each run, ensuring that your results and logs are organized.

**Usage:**

```bash
bash demo_run.sh [config_file]
```

- **`config_file` (optional):** Specifies the configuration file to use for the demo. If not provided, it defaults to `config_production.yaml`.

**Examples:**

- Run with default production configuration:
  ```bash
  bash demo_run.sh
  ```

- Run with a development configuration:
  ```bash
  bash demo_run.sh config_development.yaml
  ```

After the demo completes, a message will indicate the location of the output directory (e.g., `output_v5_YYYYMMDD_HHMMSS`), which will contain all generated results, logs, and artifacts.

### 3.2. Generating GIFs from Agent Output (`make_gif.py`)

The `make_gif.py` script automates the creation of animated GIFs from the rendered images produced by your agent. This is particularly useful for visualizing the agent's progress over time.

**Prerequisites:**

- Python 3 installed.
- `Pillow` library installed (`pip install Pillow`).

**Usage:**

Navigate to the directory containing your `output_v5_*` folders (where the `render_*.png` images are located) and run the script:

```bash
python make_gif.py
```

**Customization:**

You can adjust the following variables within `make_gif.py` to suit your needs:

- `image_folder`: The pattern to match your output folders (e.g., `output_v5_*`).
- `out_gif`: The desired filename for the generated GIF (e.g., `agent_progress.gif`).
- `frame_ms`: The duration of each frame in milliseconds (e.g., `500` for 0.5 seconds per frame).

### 3.3. Stress Testing the System (`stress_test.sh`)

The `stress_test.sh` script is designed to intentionally put the V5 system under various forms of stress to test its robustness and error handling capabilities. It includes simulations for CUDA Out-Of-Memory (OOM) errors, asset loss, and renderer crashes.

**Usage:**

```bash
bash stress_test.sh
```

**Important Notes:**

- This script is intended for testing purposes and may disrupt ongoing processes. Use with caution.
- The script includes commented-out sections for additional V5-specific stress tests, such as corrupting configuration files or simulating high I/O load. You can uncomment and modify these as needed to match your specific V5 components.
- After running the stress test, it is crucial to run your main pipeline again and thoroughly check the error handling and logs (`output_v5_*/run.log`) to assess the system's recovery.

## 4. Advanced Features and Utilities

### 4.1. Configuration Management (`config_manager.sh`)

The `config_manager.sh` script provides a convenient way to manage your V5 training and demo configurations. It allows you to list, create, and (conceptually) edit configuration files, ensuring consistency and ease of experimentation.

**Usage:**

```bash
bash config_manager.sh [command] [arguments]
```

**Commands:**

- **`list`**: Lists all available JSON configuration files in the project's `config` directory.
  ```bash
  bash config_manager.sh list
  ```

- **`use <config_file>`**: (Informational) Indicates how to use a specific configuration file with `demo_run.sh`. Note that `demo_run.sh` requires the config file to be passed as an argument.
  ```bash
  bash config_manager.sh use training_v5.json
  ```

- **`create <new_config_name>`**: Creates a new configuration file (e.g., `my_experiment.json`) by copying from a template (assumed to be `training_v5.json`).
  ```bash
  bash config_manager.sh create my_new_experiment
  ```
  This will create `my_new_experiment.json` in the config directory.

- **`edit <config_file>`**: (Instructional) Provides guidance on how to edit a specified configuration file. In a typical terminal environment, this would suggest opening the file with `nano` or `vim`. In this sandboxed environment, you would use the `file_read_text` and `file_write_text` tools.
  ```bash
  bash config_manager.sh edit my_new_experiment.json
  ```

- **`help`**: Displays the usage instructions for the `config_manager.sh` script.
  ```bash
  bash config_manager.sh help
  ```

## 5. Feedback and Troubleshooting

For any issues or to provide feedback on the V5 project or this toolkit, please refer to the `README.md` for instructions on how to submit feedback or open a GitHub issue. When reporting issues, always include the contents of the relevant `output_v5_*/run.log` file and a clear description of the steps to reproduce the problem.

## 6. Future Enhancements (Conceptual)

This toolkit provides a solid foundation. Future enhancements could include:

- **Automated Reporting:** Scripts to generate summary reports of training runs, including key metrics, performance graphs, and comparisons between different configurations.
- **Docker/Containerization Support:** Tools to easily containerize the V5 environment for consistent deployment across different machines.
- **Web-based Monitoring Dashboard:** A simple web interface to monitor training progress, visualize real-time metrics, and trigger demo runs remotely.
- **Version Control Integration:** Scripts to automatically commit and tag successful training runs with relevant metadata.
- **Automated Hyperparameter Tuning:** Integration with hyperparameter optimization frameworks to automatically find optimal training parameters.

We encourage you to explore and extend this toolkit to further enhance your development workflow!

