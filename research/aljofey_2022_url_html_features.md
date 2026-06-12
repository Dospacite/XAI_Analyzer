# An Effective Detection Approach for Phishing Websites Using URL and HTML Features

## General information

- Authors: Ali Aljofey, Qingshan Jiang, Abdur Rasool, Hui Chen, Wenyin Liu, Qiang Qu, Yang Wang, et al.
- Year: 2022
- Venue: Scientific Reports, volume 12, article 8842
- DOI: https://doi.org/10.1038/s41598-022-10841-5
- Main modalities: URL, HTML source, DOM hyperlinks, page text, form metadata
- Dataset: 60,252 webpages: 32,972 benign and 27,280 phishing, plus evaluation on an existing benchmark dataset.

## Used model

The paper evaluates multiple classifiers and reports the best result with XGBoost on the combined URL, text, hyperlink, and form feature vector. It also evaluates Random Forest, Logistic Regression, Naive Bayes, and an ensemble of Random Forest plus AdaBoost.

## Extracted features

The paper defines 15 feature groups, F1 through F15:

- F1: URL character sequence features. The URL is tokenized at character level, mapped to integer tokens, and padded/truncated to length 200.
- F2: Textual content features from HTML. Character-level TF-IDF is extracted from plaintext and noisy HTML attribute values, especially values around `div`, `h1`, `h2`, `body`, and `form` tags after removing JavaScript, CSS, punctuation, and numbers.
- F3: Ratio of JavaScript files to total hyperlinks, from `script src` links.
- F4: Ratio of CSS files to total hyperlinks, from `link href` links.
- F5: Ratio of image files to total hyperlinks, from `img src` links.
- F6: Ratio of anchor files to total hyperlinks, from `a href` links.
- F7: Ratio of anchor tags without an `href` attribute to total hyperlinks.
- F8: Ratio of null or empty hyperlinks to total hyperlinks, including values such as `#`, page fragments, and `javascript:void(0)`.
- F9: Total number of hyperlinks in the webpage.
- F10: Ratio of internal hyperlinks to total hyperlinks.
- F11: Ratio of external hyperlinks to total hyperlinks.
- F12: Ratio of external hyperlinks to internal hyperlinks.
- F13: Ratio of invalid hyperlinks to total hyperlinks.
- F14: Total number of forms in the webpage.
- F15: Ratio of suspicious forms to total forms. A form is suspicious when the `action` URL is null, invalid, or points to an external domain.

## Methodology

The method crawls the page, parses HTML with BeautifulSoup, extracts a DOM tree, generates URL character-sequence vectors, HTML text TF-IDF vectors, and a 13-dimensional hyperlink/form vector. These vectors are concatenated and passed to machine learning classifiers. The paper emphasizes client-side features that do not depend on WHOIS, DNS, search-engine indexing, PageRank, or web traffic services.

## Sources

- Paper page: https://www.nature.com/articles/s41598-022-10841-5
- DOI: https://doi.org/10.1038/s41598-022-10841-5
