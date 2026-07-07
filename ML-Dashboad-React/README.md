# React ML Dashboard 🚀

<!-- <div align="center">
<img width="1200" height="475" alt="CodeVeda ML Dashboard Banner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div> -->

An interactive machine learning learning-studio built with **React**, **Vite**, **TypeScript**, and **Tailwind CSS**. The dashboard lets you explore classic ML workflows in the browser with a polished UI, animated transitions, and live visualizations.

## ✨ Overview

This project is designed as a client-side training and demo environment for common machine learning concepts. It includes multiple modules grouped into two levels:

- **Level 02**: foundational ML workflows
- **Level 03**: more advanced classification, regression, and deep learning demos

Everything runs locally in the browser. There is no backend service required for the React app.

## 🔥 Features

- 📊 Interactive dashboards for multiple ML algorithms
- 🌈 Smooth animated transitions using Motion
- 🎛️ Algorithm tabs with clear model descriptions and level labels
- 📁 CSV parsing and dataset handling with PapaParse
- 📈 Data visualizations powered by Recharts
- 🎨 Modern UI styling with Tailwind CSS
- 🧠 Machine learning concept exploration directly in the browser
- ⚡ Fast development workflow powered by Vite

## 🧰 Tech Stack

- **Frontend:** React 18, TypeScript
- **Bundler / Dev server:** Vite
- **Styling:** Tailwind CSS 4, PostCSS, Autoprefixer
- **Animation:** Motion
- **Charts:** Recharts
- **Icons:** Lucide React
- **CSV handling:** PapaParse

## 🧪 Included ML Modules

- 🪓 **Decision Tree Classifier**
   - Fisher Iris dataset exploration
   - Interactive tree training and predictions

- 🧭 **K-Means & DBSCAN**
   - Unsupervised clustering visualizations
   - PCA projection, silhouette analysis, elbow curve

- 🌳 **Random Forest Regressor**
   - Regression workflow with outlier handling
   - Cross-validation and residual analysis

- 🧠 **Deep Neural Network**
   - Feed-forward deep learning demo
   - Training convergence visualization

- 🧮 **Support Vector Machine**
   - Linear and RBF classifier comparison
   - Multiclass classification visualization

## 📁 Project Structure

```text
ML-Dashboad-React/
├── app.py
├── index.html
├── package.json
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── index.css
│   ├── types.ts
│   └── components/
│       ├── DecisionTree.tsx
│       ├── KMeansClustering.tsx
│       ├── NeuralNetwork.tsx
│       ├── RandomForest.tsx
│       └── SvmSentiment.tsx
├── datasets/
└── pages/
```

## 📦 Datasets

The dashboard references local CSV datasets stored in the `datasets/` folder. Common examples include:

- Iris dataset
- Sentiment dataset
- House prediction dataset

If you add or replace datasets, keep the filenames consistent with the component logic.

## ✅ Prerequisites

- Node.js 18 or newer
- npm

## 🚀 Getting Started

### 1) Install dependencies

```bash
npm install
```

### 2) Start the development server

```bash
npm run dev
```

### 3) Open the app

By default, Vite serves the project at:

```text
http://localhost:3000
```

## 🛠️ Available Scripts

- `npm run dev` - start the local development server on port 3000
- `npm run build` - create a production-ready build
- `npm run preview` - preview the production build locally on port 3000

## 🧭 How to Use

1. Open the dashboard in your browser.
2. Select an ML algorithm from the top navigation cards.
3. Adjust inputs, parameters, or dataset options inside the active module.
4. Review charts, metrics, and predictions in real time.

## 📌 Notes

- The app is fully client-side; no API key is required for the React dashboard.
- The UI is optimized for interactive learning and demonstrations rather than production inference.
- Some model demos rely on the expected dataset filenames and shapes. If you rename CSV files, update the corresponding component references.

## 🧱 Build Output

When you run `npm run build`, Vite generates an optimized production bundle suitable for deployment on static hosting platforms such as:

- Vercel
- Netlify
- GitHub Pages
- Any static file server

## 🐞 Troubleshooting

- If the app does not start, confirm that Node.js is installed and that `npm install` completed successfully.
- If a chart or dataset fails to load, verify the CSV file exists in `datasets/` and matches the expected name.
- If the browser shows a blank page after changes, check the terminal for TypeScript or Vite build errors.

## 📜 License

No license file is currently included. Add one if you plan to share or publish the project publicly.

## 🙌 Acknowledgements

Built as part of the CodeVeda machine learning dashboard experience using modern web tooling and interactive visualization libraries.
