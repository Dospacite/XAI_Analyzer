# Look Before You Leap: Detecting Phishing Web Pages by Exploiting Raw URL and HTML Characteristics

## General information

- Authors: Chidimma Opara, Yingke Chen, Bo Wei
- First arXiv submission: 2020
- Journal version: 2024
- Venue: Expert Systems with Applications, volume 236, article 121183
- arXiv: https://arxiv.org/abs/2011.04412
- DOI: https://doi.org/10.1016/j.eswa.2023.121183
- Main modalities: Raw URL and raw HTML content

## Used model

The proposed WebPhish model is an end-to-end deep neural network. It uses character embeddings for URL and HTML inputs, concatenates the embedding matrices, and applies convolutional layers for classification.

## Extracted features

No hand-crafted URL, HTML, metadata, or domain features are extracted. The paper is designed to avoid manual lexical/statistical feature engineering.

## Raw data and methodology

- Raw input 1: URL string.
- Raw input 2: HTML content of the webpage.
- Encoding: Characters from URLs and HTML are embedded into dense vectors.
- Fusion: URL and HTML embedding matrices are concatenated.
- Learning: Convolutional layers learn semantic character patterns across the fused raw representation.
- Output: The model classifies each webpage as phishing or benign.

The paper reports 98.1% accuracy on real-world phishing data and positions the model as a way to capture URL and HTML semantics without manually curated phishing features.

## Sources

- arXiv page: https://arxiv.org/abs/2011.04412
- arXiv DOI: https://doi.org/10.48550/arXiv.2011.04412
- Journal DOI: https://doi.org/10.1016/j.eswa.2023.121183
