# MACS40123_final
This repository contains the key scripts, data processing utilities, statistical models, and result files used in this study to reproduce and verify the causal inference and exploratory analyses. Below is an overview of the main files and directories, along with direct links.

## Visualization and Causal Analyses
- [ITS Analysis](/Visualization/its.py): Interrupted Time Series (ITS) model to assess the long-term effect of the NSL intervention on the Vague Expression Ratio.
- [DID Analysis](/Visualization/did.py): Difference-in-Differences (DID) approach comparing observed data with counterfactual predictions to strengthen causal inference.
- [Distribution Graph](/Visualization/distribution_graph.py): Scripts for generating distribution plots and visualizing trends in key indicators.
- [Prophet Time Series Model](/Visualization/prophet.py): Uses Prophet to generate predictions under a no-intervention scenario for comparison.

## Counterfactual Data
- [Counterfactual Data CSV](/Counterfactual%20Data/Counter_factual_data.csv): Dataset containing counterfactual predictions.
- [Prophet Training Script](/Counterfactual%20Data/prophet.py): Used to train the Prophet model on pre-intervention data.

## Data Processing
- [Extract Titles and Content](/Data%20Processing/Data%20cleaning/extract_title_content.py): Extracts and preprocesses post titles and content.
- [Data Cleaning Script](/Data%20Processing/Data%20cleaning/new%20cleaned.py): Cleans and formats raw data for further analysis.
- [Data Annotation Script](/Data%20Processing/Data_annotation/Data_annotation.py): Annotates data for sentiment, vagueness, and government criticism.
- [Data Segmentation](/Data%20Processing/Data_segment/segment_all.py): Segments processed data for modeling and analysis.

## Exploratory Analysis and Frequent Itemsets
- [Frequent Itemsets Main Script](/Exploratory%20work/find_frequent_itemsets/find_frequent_itemsets.py): Identifies frequent co-occurring terms and patterns within the dataset.
- [Itemsets Script](/Exploratory%20work/find_frequent_itemsets/local_itemsets.py): Processes itemsets locally and supports the main script.
- [Itemset Co-occurrence Results](https://github.com/hchen0628/MACS40123-Lab1/tree/main/itemsets): Contains JSON files of co-occurrence analyses for different themes (e.g., authoritarianism, democracy, government).
- [K-Means Clustering](/Exploratory%20work/Kmeans/): K-Means clustering scripts, results, and cluster centers.
- [PCA and SVD Results](/Exploratory%20work/PCA/): Scripts and JSON files showing principal component analysis results and top features extracted.

Please refer to the main documentation and in-code comments for detailed explanations of data processing steps, model training procedures, analytical frameworks, and interpretation of the results.
