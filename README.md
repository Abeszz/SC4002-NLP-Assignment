# SC4002 - Natural Language Processing Assignment

Welcome to the SC4002 - Natural Language Processing Assignment repository! This project includes a `.ipynb` file that can be run on your local machine using Visual Studio Code, or it can be uploaded and run on Google Colab.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
  - [Running on Visual Studio Code](#running-on-visual-studio-code)
  - [Running on Google Colab (Recommended)](#running-on-google-colab-recommended)
- [Output](#output)


---

## Prerequisites

- Python 3.x (Python 3.9 or later is recommended)

Ensure that you have Python 3 installed on your machine before proceeding with the installation steps. You can check your Python version by running the following command:

```bash
python --version
```

## Installation

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/Abeszz/SC4002-NLP-Assignment.git
cd SC4002-NLP-Assignment
```

---

## Running the Notebook

You can run the `.ipynb` file using one of the following methods:

## Running on Visual Studio Code

1. Open Visual Studio Code and navigate to the project folder.
2. Ensure the **Jupyter** extensions are installed in VS Code.
   - To install these, go to the **Extensions** sidebar (usually on the left) and search for "Jupyter."
   - Install Jupyter extension if is not already installed.
3. Open the `SC4002.ipynb` file in VS Code by navigating to **File** > **Open File** and selecting the notebook file.
4. Once the notebook is open, select a Python interpreter or kernel if prompted. This will allow you to run the code cells.
5. Run each cell by clicking the "Play" icon next to each cell, or use the **Run All** option to execute all cells sequentially.

## Running on Google Colab (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload the `SC4002.ipynb` file to Colab using one of the following options:
   - **Option 1**: Click on **File** > **Upload notebook** and select the `SC4002.ipynb` file from your local machine.
   - **Option 2**: Click **File** > **Open notebook**, go to the **GitHub** tab, and paste the repository URL to open the notebook directly from GitHub.
3. Once the file is loaded into Colab, connect to a runtime environment by clicking on the **Connect** button in the top-right corner.
4. Run each cell by clicking the "Play" icon next to each cell, or select **Runtime** > **Run all** to execute the entire notebook.

> **Note**: Google Colab provides free GPU/TPU support if required for your project. You can enable it by going to **Runtime** > **Change runtime type** and selecting **GPU** or **TPU**.

## Output

Each method in this notebook will output its results to a CSV file. The generated CSV files will be saved in the project directory. The CSV files will be named according to the method used, for example: `part3_blstm_bgru_results.csv`.

The CSV files will include:
- **Best Hyperparameters Chosen**: Only the optimal hyperparameter values selected during the run.
- **Best Result**: The best result or metric achieved with the chosen hyperparameters.

