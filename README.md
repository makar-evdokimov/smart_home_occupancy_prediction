# Predicting House Occupancy in a Smart Home Environment

A multi-model agent that predicts hourly probabilities of a home being occupied for the next 24 hours, based solely on energy consumption data from smart electricity meters.

## Abstract

With rising global temperatures and weather events getting more extreme everywhere, the reduction of CO2 emissions should be a primary goal for everybody. Providing home owners with recommendations on how to use heat, A/C and electricity more efficiently can greatly contribute to this goal. One important basis for these recommendations is information on when the home will be occupied. This work provides an agent that can predict hourly probabilities of a home being occupied for the next 24 hours based on energy consumption data provided by smart electricity meters, using various supervised learning algorithms. It is also capable of inferring the occupancy labels, and thus can make predictions of future occupancy with or without these labels being available alongside the consumption data. This helps to lower the threshold for making a home "smarter" since it does not require any additional intrusive sensors beyond smart electricity meters.

## Approach

* Occupancy is inferred and predicted using several supervised ML algorithms (SVM, random forest, LightGBM), with the best-performing model selected for each task.
* A custom genetic algorithm is used for feature engineering, evolving new features on top of manually engineered baseline features. It is implemented as the [`gp_features`](gp_features/) Python package.
* Data is aggregated and cleaned from three public smart-meter datasets: SMART*, ECO, and DRED.
* The full methodology, exploratory analysis, model development, and evaluation (including feature importance / explainability) are documented in [`main.ipynb`](main.ipynb).

## Repository contents

- `main.ipynb` — the main notebook: data aggregation, feature engineering, model training/evaluation, and the multi-model prediction agent.
- `gp_features/` — the genetic feature-engineering package used by the notebook.
- `images/` — figures referenced from the notebook.
- `requirements.txt` — conda environment spec used to reproduce the results (`conda create --name <env> --file requirements.txt`).

## Running it

The notebook is formatted to be opened in Colab, where large pieces of code can be scrolled/collapsed.

To run it, place the `gp_features/` folder in your working directory alongside the raw dataset files, which are available via this Google Drive folder:
https://drive.google.com/drive/folders/1Or4cOauI3TEHEVMr2xilQ7Uk1A63Dsfr?usp=sharing
