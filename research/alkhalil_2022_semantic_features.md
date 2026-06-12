# Phishing Website Detection With Semantic Features Based on Machine Learning Classifiers

## General information

- Authors: Ammar Almomani, Mohammad Alauthman, Mohd Taib Shatnawi, Mohammed Alweshah, Ayat Alrosan, Waleed Alomoush, Brij B. Gupta
- Year: 2022
- Venue: International Journal on Semantic Web and Information Systems, volume 18, issue 1
- DOI: https://doi.org/10.4018/IJSWIS.297032
- Main modalities: URL, domain identity, abnormal URL/page behavior, HTML, JavaScript, domain metadata, search/traffic metadata

## Used model

The paper compares 16 machine learning classifiers for phishing website detection. The accessible abstract reports GradientBoostingClassifier and RandomForestClassifier as the best-performing classifiers, at about 97% accuracy.

## Extracted features

The article describes semantic features grouped into URL and domain identity, abnormal features, HTML/JavaScript features, and domain-based features. The feature set reported in the accessible article text and table snippets includes:

- Using the IP address.
- Long URL to hide the suspicious part.
- URL shortening service, such as TinyURL.
- URL having `@` symbol.
- Redirecting using `//`.
- Prefix or suffix separated by `-` in the domain.
- Subdomain and multi-subdomain.
- HTTPS / SSL final state.
- Domain registration length.
- Favicon.
- Non-standard port.
- HTTPS token in the domain part of the URL.
- Request URL.
- URL of anchor.
- Links in `meta`, `script`, and `link` tags.
- Server Form Handler (SFH).
- Submitting information to email.
- Abnormal URL.
- Website forwarding / redirect count.
- `onMouseOver` status-bar manipulation.
- Disabled right click.
- Popup window.
- IFrame.
- Age of domain.
- DNS record.
- Website traffic.
- PageRank.
- Google index.
- Number of links pointing to the page.
- Statistical report / blacklist-style reputation signal.

The article also reports a second numeric-feature dataset with overlapping low-level URL and page-content features:

- Number of dots.
- Subdomain level.
- Path level.
- URL length.
- Number of dashes.
- Number of dashes in hostname.
- At symbol.
- Tilde symbol.
- Number of underscores.
- Number of percent signs.
- Number of query components.
- Number of ampersands.
- Number of hash signs.
- Number of numeric characters.
- No HTTPS.
- Number of sensitive words.
- Embedded brand name.
- Percentage of external hyperlinks.
- Percentage of external resource URLs.
- External favicon.
- Insecure forms.
- Relative form action.
- External form action.
- Abnormal form action.
- Percentage of null/self-redirect hyperlinks.
- Frequent domain-name mismatch.
- Fake link in status bar.
- Right click disabled.
- Popup window.
- Submit information to email.

## Methodology

The workflow extracts semantic website attributes from URL/domain identity, HTML and JavaScript behavior, abnormal form/link behavior, and external domain/reputation metadata. It then evaluates multiple machine learning classifiers and compares their accuracy across the extracted feature sets.

## Sources

- ScienceDirect article page: https://www.sciencedirect.com/org/science/article/pii/S155262832200045X
- Bibliographic metadata: https://colab.ws/articles/10.4018%2FIJSWIS.297032
- DOI: https://doi.org/10.4018/IJSWIS.297032
