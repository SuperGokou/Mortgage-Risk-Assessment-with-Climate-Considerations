# Mortgage Risk Assessment: From Extreme Weather Perspective

## Team Members
- Yao Zhang
- Maggie Chen
- Vincent Stone
- Ming Xia

## Institution
Harvard University

## Date
July 2024

## Abstract
This project addresses the significant challenges posed by climate risk to the mortgage industry, exacerbated by the increasing frequency and severity of climate-related disasters. Using data from PNC Bank, this capstone aims to identify and quantify climate-related risks on mortgage portfolios by integrating historical mortgage data with economic indicators and natural disaster information. The goal is to enhance underwriting processes, mitigate risks, and ensure financial stability through advanced predictive models.

## Problem Statement
Climate risk presents significant challenges to the mortgage lending industry due to the increasing frequency and severity of climate-related disasters such as hurricanes, wildfires, floods, and heatwaves. This project focuses on a single type of weather event in a specific region to provide an in-depth analysis of climate-related risks to mortgage portfolios.

## Objectives
- To identify and quantify the impact of climate-related risks on mortgage portfolios.
- To integrate climate risk into mortgage underwriting processes.
- To develop predictive models for mortgage default and prepayment risks in the context of climate change.

## Methodology
- **Data Collection:** Merge historical mortgage data and economic indicators with weather and natural disaster information.
- **Predictive Modeling:** Employ advanced predictive models such as the additive Cox proportional hazards model, generalized additive logistic models, and Bayesian methods.
- **Data Analysis:** Use statistical and machine learning techniques to analyze and predict risks.

## Expected Outcomes
The findings will provide actionable insights for financial institutions to improve their underwriting processes and manage risks effectively. This approach could potentially extend to consumer-level risk assessments, enhancing financial stability and resilience against climate-induced financial disruptions.

## Dataset Description
- **Brixia Score COVID-19 Dataset:** 4,695 CXR images from March 4th to April 4th, 2020, for COVID-19 positive cases.
- **NIH Chest X-ray Dataset:** 112,120 images with pathology labels, utilized for negative COVID-19 instances.
- **CovidGR Dataset:** Aimed at evaluating machine learning models for COVID-19 detection from X-ray images.

## Data Preprocessing and Augmentation
- **Standardization and Preprocessing:** Adjust image size, enhance contrast, and apply noise reduction to normalize images across datasets.
- **Data Augmentation:** Implement rotation, flipping, scaling, and cropping to increase dataset variability and combat overfitting.

## Addressing Data Imbalance
- **Adding Data:** Augment the dataset with additional positive COVID-19 X-ray samples.
- **Class Weights:** Use computed class weights to adjust the focus during model training.
- **Oversampling and Undersampling:** Employ techniques like SMOTE to enhance balance.

## License
This project is licensed under the MIT License. See `LICENSE` for more information.
