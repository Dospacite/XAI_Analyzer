# Machine Learning Based Phishing Detection from URLs

## General information

- Authors: Ozgur Koray Sahingoz, Ebubekir Buber, Onder Demir, Banu Diri
- Year: 2019
- Venue: Expert Systems with Applications, volume 117, pages 345-357
- DOI: https://doi.org/10.1016/j.eswa.2018.09.029
- Main modalities: URL lexical structure, domain tokens, subdomain/path tokens, brand and keyword lists
- Dataset: 73,575 URLs, including 36,400 legitimate URLs and 37,175 phishing URLs, according to the paper metadata and accessible text.

## Used model

The paper evaluates Decision Tree, AdaBoost, K-Star, k-nearest neighbor, Random Forest, Sequential Minimal Optimization/SVM, and Naive Bayes. It reports Random Forest with NLP-based URL features as the best-performing configuration.

## Extracted features

The paper groups features into NLP-based features, word-vector features, and hybrid features. The accessible feature table lists these NLP-based URL features:

- Raw word count after parsing the URL by special characters.
- Brand check for domain.
- Average word length in the raw URL word list.
- Longest word length in the raw URL word list.
- Shortest word length in the raw URL word list.
- Standard deviation of word lengths in the raw URL word list.
- Adjacent word count from the word-decomposition module.
- Average adjacent word length.
- Separated word count after decomposing adjacent words.
- Keyword count in the URL.
- Brand name count in the URL.
- Similar keyword count.
- Similar brand name count.
- Random word count.
- Target brand name count.
- Target keyword count.
- Other words count, for words in a dictionary but not in brand or keyword lists.
- Digit count in the domain.
- Digit count in the subdomain.
- Digit count in the file path.
- Subdomain count.
- Random domain indicator.
- Domain length.
- Subdomain length.
- Path length.
- Known TLD indicator.
- `www` occurrence feature.
- `com` occurrence feature.
- Punycode indicator.
- Special-character count for `-`.
- Special-character count for `.`.
- Special-character count for `/`.
- Special-character count for `@`.
- Special-character count for `?`.
- Special-character count for `&`.
- Special-character count for `=`.
- Special-character count for `_`.

The word-vector feature set represents URL tokens as word vectors. The hybrid setting combines the NLP-based features with the word-vector representation.

## Methodology

The system parses URLs into words and structural segments, computes NLP-style lexical statistics, compares words against brand and phishing keyword lists, detects random-looking tokens/domains, and trains standard machine learning classifiers. It does not require visiting the webpage or using third-party services, so it is intended for real-time URL-only phishing detection.

## Sources

- DOI landing page: https://doi.org/10.1016/j.eswa.2018.09.029
- Metadata and abstract: https://colab.ws/articles/10.1016%2Fj.eswa.2018.09.029
- Feature-table text surfaced in indexed full-text preview: https://www.researchgate.net/publication/344952543_Machine_learning_based_phishing_detection_from_URLs
