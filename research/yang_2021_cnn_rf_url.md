# Phishing Website Detection Based on Deep Convolutional Neural Network and Random Forest Ensemble Learning

## General information

- Authors: Rundong Yang, Kangfeng Zheng, Bin Wu, Chunhua Wu, Xiujuan Wang
- Year: 2021
- Venue: Sensors, 21(24), article 8281
- DOI: https://doi.org/10.3390/s21248281
- Main modalities: URL string only

## Used model

The paper combines convolutional neural networks with random forest classifiers. CNNs extract multilevel URL representations, several Random Forest classifiers classify those multilevel features, and a winner-take-all strategy produces the final prediction.

## Extracted features

No hand-crafted phishing features are extracted. The method explicitly avoids third-party features such as PageRank, search-engine indexing, traffic measurement, and domain age.

## Raw data and methodology

- Raw input: URL string.
- Encoding: Character embedding converts each URL into a fixed-size matrix.
- Feature learning: CNN layers automatically extract multilevel URL features.
- Classification: Multiple Random Forest classifiers classify learned CNN feature levels.
- Decision fusion: Winner-take-all combines classifier outputs.

The design is URL-only and does not access webpage HTML, WHOIS, DNS, or traffic services.

## Sources

- Paper page: https://www.mdpi.com/1424-8220/21/24/8281
- DOI: https://doi.org/10.3390/s21248281
