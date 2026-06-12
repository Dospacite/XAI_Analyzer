# HTMLPhish: Enabling Phishing Web Page Detection by Applying Deep Learning Techniques on HTML Analysis

## General information

- Authors: Chidimma Opara, Bo Wei, Yingke Chen
- Year: 2020
- Venue: International Joint Conference on Neural Networks (IJCNN) 2020
- arXiv: https://arxiv.org/abs/1909.01135
- DOI: https://doi.org/10.1109/IJCNN48605.2020.9207707
- Main modalities: Raw HTML document content
- Dataset: More than 50,000 HTML documents with a real-world phishing/benign distribution.

## Used model

The paper uses a convolutional neural network over HTML document embeddings. It combines word-level and character-level embeddings before CNN-based classification.

## Extracted features

No manual features are extracted. The paper explicitly avoids extensive hand-crafted feature engineering.

## Raw data and methodology

- Raw input: HTML document text/content of each webpage.
- Representation: HTML is transformed into word and character embeddings.
- Fusion: Word and character embeddings are concatenated.
- Learning: CNN layers learn semantic dependencies in HTML textual content.
- Classification: The learned representation is used to classify pages as phishing or benign.

The method is client-side and language independent because it learns from raw HTML tokens rather than fixed English keyword rules or external metadata.

## Sources

- arXiv page: https://arxiv.org/abs/1909.01135
- arXiv DOI: https://doi.org/10.48550/arXiv.1909.01135
- IJCNN DOI: https://doi.org/10.1109/IJCNN48605.2020.9207707
