# PhishSim: Aiding Phishing Website Detection with a Feature-Free Tool

## General information

- Authors: Rizka Purwanto, Arindam Pal, Alan Blair, Sanjay Jha
- Year: 2022
- Venue: IEEE Transactions on Information Forensics and Security, volume 17, pages 1497-1512
- arXiv: https://arxiv.org/abs/2207.10801
- DOI: https://doi.org/10.1109/TIFS.2022.3164212
- Main modalities: Raw HTML webpage content

## Used model

PhishSim uses Normalized Compression Distance (NCD) for similarity-based classification. It also uses Furthest Point First to select phishing prototypes and an incremental learning framework for continuous adaptation.

## Extracted features

No explicit features are extracted. The paper describes the method as feature-free and parameter-free with respect to website feature design.

## Raw data and methodology

- Raw input: HTML content of webpages.
- Similarity metric: Normalized Compression Distance, computed by compressing webpage representations.
- Prototype selection: Furthest Point First selects representative phishing webpage prototypes.
- Classification: A candidate webpage is compared against known phishing prototypes using compression-based similarity.
- Adaptation: Incremental learning allows the system to update over time without designing new features for concept drift.

The method removes dependence on a fixed URL, HTML, or domain feature set and instead uses whole-page HTML similarity.

## Sources

- arXiv page: https://arxiv.org/abs/2207.10801
- arXiv DOI: https://doi.org/10.48550/arXiv.2207.10801
- IEEE DOI: https://doi.org/10.1109/TIFS.2022.3164212
