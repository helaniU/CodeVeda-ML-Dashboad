# CodeVeda ML Project 🚀

This workspace contains two machine learning dashboard implementations built for learning, demos, and experimentation:

- 🌐 **React ML Dashboard** in [ML-Dashboad-React/](ML-Dashboad-React/)
- 🧪 **Streamlit Dashboard** in [Streamlit Dashboard/](Streamlit%20Dashboard/)

Both projects focus on classic ML workflows, interactive visualizations, and approachable user experiences for exploring model behavior.

## ✨ Project Highlights

- 📊 Interactive ML dashboards and visual analytics
- 🧠 Classic machine learning workflows
- 🎨 Clean, modern UI presentation with icons and visual polish
- 📁 Local datasets for hands-on experimentation
- ⚡ Separate web stacks for frontend and Python-based exploration

## 🧰 Tech Stack Overview

### 🌐 React ML Dashboard

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Motion
- Recharts
- Lucide React
- PapaParse

### 🧪 Streamlit Dashboard

- Streamlit
- Python 3
- pandas
- NumPy
- Matplotlib
- scikit-learn

## 📁 Workspace Structure

```text
CodeVeda ML Project/
├── ML-Dashboad-React/
│   ├── app.py
│   ├── package.json
│   ├── src/
│   ├── datasets/
│   ├── pages/
│   └── README.md
├── Streamlit Dashboard/
│   ├── app.py
│   ├── datasets/
│   ├── Level_02/
│   ├── Level_03/
│   ├── pages/
│   └── Readme.md
└── README.md
```

## 🚀 Quick Start

Choose the dashboard you want to run, then follow its setup steps below.

## 🌐 React ML Dashboard

This app is a browser-based ML studio built with React and Vite.

### Prerequisites

- Node.js 18 or newer
- npm

### Install

```bash
cd "ML-Dashboad-React"
npm install
```

### Run

```bash
npm run dev
```

The app typically runs at:

```text
http://localhost:3000
```

### Build

```bash
npm run build
```

### Preview the production build

```bash
npm run preview
```

### What it includes

- Decision Tree classifier
- K-Means and DBSCAN clustering
- Random Forest regression
- Deep neural network demo
- SVM sentiment classification

## 📦 Datasets

Both dashboards use local datasets stored in their respective `datasets/` folders. Common examples include:

- Iris dataset
- Sentiment dataset
- House prediction dataset
- Stock prices dataset

If you replace a dataset, keep the filenames and expected column structure aligned with the scripts and components that load them.

## 🧭 Development Notes

- The React and Streamlit apps are independent and can be run separately.
- Each app has its own README with deeper project-specific documentation.
- Some scripts use relative paths, so run commands from the correct folder.

## 🐞 Troubleshooting

- If a React command fails, confirm you are inside `ML-Dashboad-React/` and that `npm install` completed successfully.
- If a Streamlit command fails, confirm the Python environment is activated and the required packages are installed.
- If datasets fail to load, verify the CSV file names and folder paths.
- If the browser shows a blank page or runtime error, inspect the terminal output for missing dependencies or type errors.

## 🚀 Deployment

- The React dashboard can be deployed to static hosting platforms such as Vercel, Netlify, or GitHub Pages.
- The Streamlit dashboard can be deployed to Streamlit Community Cloud, a server, or a containerized environment.

## 👤 Author / Maintainer

- Helani Umesha Ambalangodage (Me)

## 📜 License

No license file is included at the root level. Add one if you plan to publish or distribute the workspace publicly.

## 🙌 Acknowledgements

Built as a practical learning workspace for machine learning visualization, experimentation, and dashboard design.

