# Streamlit Dashboard 🌿

An interactive **Streamlit** machine learning dashboard for exploring classic ML workflows, model training, and visual analysis. This folder contains the Streamlit app shell, multipage UI assets, and standalone ML experiment scripts used for the dashboard content.

## ✨ Overview

The Streamlit dashboard is built for fast experimentation and model storytelling. It focuses on approachable, visual machine learning demos such as:

- 🌲 Decision Tree classification on the Iris dataset
- 🌼 K-Means and DBSCAN clustering analysis
- 🏠 Random Forest regression for house price prediction
- 💬 Support Vector Machine sentiment classification
- 🧠 Additional preprocessing and ML workflow experiments

The app is lightweight and runs locally from the command line with Streamlit.

## 🔥 Key Features

- 📱 Clean multipage Streamlit interface
- 📊 Interactive charts and model visualizations
- 🧮 Real ML training and evaluation with scikit-learn
- 📁 Local CSV and built-in dataset support
- 🎛️ Sidebar controls for model hyperparameters
- 🎨 Custom UI styling with HTML/CSS inside Streamlit
- ⚡ Fast local iteration for demos, prototypes, and learning

## 🧰 Tech Stack

- **Framework:** Streamlit
- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** scikit-learn
- **Datasets:** Iris, sentiment text data, house price data

## 🛠️ Tools and Libraries

The dashboard scripts use the following core packages:

- `streamlit` for the UI
- `pandas` for tabular data manipulation
- `numpy` for numerical operations
- `matplotlib` for plots
- `scikit-learn` for training, evaluation, and preprocessing

## 📁 Project Structure

```text
Streamlit Dashboard/
├── app.py
├── Readme.md
├── datasets/
├── Level_02/
│   ├── Decision_Tree_Iris.py
│   └── KMeans_Iris.py
├── Level_03/
│   ├── Neural_Network_MNIST.py
│   ├── Random_Forest_House.py
│   └── SVM_sentiment.py
├── pages/
│   ├── decision_tree.py
│   ├── kmeans.py
│   ├── neural_network.py
│   ├── random_forest.py
│   └── svm.py
└── ml_env/
```

## 📦 Datasets

This project includes local data files in the `datasets/` folder. Examples include:

- Iris dataset
- Sentiment dataset
- House prediction dataset
- Stock prices dataset

Some of the standalone scripts reference these files directly, so keep the dataset names consistent if you replace them.

## ✅ Prerequisites

- Python 3.10+ recommended
- pip
- Streamlit installed in your environment

## 🚀 Getting Started

### 1) Create or activate your Python environment

If you already have the bundled virtual environment, activate it from the project root.

On Windows:

```bash
ml_env\Scripts\activate
```

If you want to use your own environment, create one first and activate it.

### 2) Install dependencies

There is no `requirements.txt` file in the folder, so install the needed packages manually:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### 3) Run the dashboard

From the `Streamlit Dashboard` folder, start the app with:

```bash
streamlit run app.py
```

## 🌐 What Opens

Streamlit will launch a local development server, usually at:

```text
http://localhost:8501
```

Open that address in your browser to use the dashboard.

## 🧭 How to Use

1. Launch the app with `streamlit run app.py`.
2. Use the sidebar to navigate or adjust model controls.
3. Open the desired page or demo from the Streamlit sidebar.
4. Review plots, metrics, predictions, and model behavior.
5. Experiment with hyperparameters to compare outcomes.

## 🧪 Included ML Demos

- 🌲 **Decision Tree Iris**
	- Classification with train/test split
	- Accuracy, confusion matrix, and tree visualization

- 🌼 **K-Means vs DBSCAN**
	- Feature scaling, clustering comparison, and PCA projection
	- Elbow method and silhouette score analysis

- 🏠 **Random Forest House Prediction**
	- Outlier removal and feature engineering
	- Regression metrics, feature importance, and tuning

- 💬 **SVM Sentiment Analysis**
	- Text vectorization with TF-IDF
	- Linear and RBF SVM comparison

- 🧠 **Neural Network Demo**
	- Deep learning exploration for the advanced level section

## 🧩 Notes About the Code

- `app.py` currently acts as the Streamlit launch point.
- The `pages/` folder is Streamlit-compatible and can be used for multipage navigation.
- The `Level_02/` and `Level_03/` scripts are standalone ML examples and can also be run directly for experimentation.
- Several scripts use relative dataset paths, so run them from the project folder to avoid file-not-found errors.

## 🧱 Recommended Workflow

For the smoothest experience:

1. Activate the Python environment.
2. Install the dependencies listed above.
3. Run the dashboard with Streamlit.
4. Keep the `datasets/` folder in place so the examples continue to work.

## 🐞 Troubleshooting

- If Streamlit is not recognized, confirm the virtual environment is activated.
- If a CSV file is missing, verify the `datasets/` folder still contains the expected filename.
- If a chart fails to render, check the terminal for Python exceptions or missing packages.
- If you see import errors, reinstall the libraries with `pip install streamlit pandas numpy matplotlib scikit-learn`.

## 🚀 Deployment

The dashboard can be deployed to any environment that supports Streamlit applications, including:

- Streamlit Community Cloud
- A local VM or server
- Docker-based hosting

Before deploying, make sure your environment includes all required Python packages and that the dataset paths are valid.

## 📜 License

No license file is included in this folder. Add one if you plan to publish or distribute the project publicly.

## 🙌 Acknowledgements

Built as part of the CodeVeda machine learning learning experience using Streamlit and the Python data science stack.
