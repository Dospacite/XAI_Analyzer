# HinPhish: An Effective Phishing Detection Approach Based on Heterogeneous Information Networks

## General information

- Authors: Bingyang Guo, Yunyi Zhang, Chengxi Xu, Fan Shi, Yuwei Li, Min Zhang
- Year: 2021
- Venue: Applied Sciences, 11(20), article 9733
- DOI: https://doi.org/10.3390/app11209733
- Main modalities: HTML source, hyperlinks, domain relationships, resource links

## Used model

HinPhish is built around a heterogeneous information network (HIN) and a modified authority-ranking algorithm. It computes HinPhish scores for domain and resource nodes, then feeds the resulting node attributes or score matrix into machine learning classifiers for website classification.

## Extracted features

HinPhish does not use a flat hand-crafted lexical feature table like URL-length or WHOIS-age systems. Its extracted feature representation is graph based. The paper extracts the following raw objects and derived relationship features:

- HTML source code for each website.
- Hyperlinks from `a` tags.
- Image resource links from `img src`.
- CSS/resource links from `link` tags.
- External script links from `script src`.
- Domain nodes, representing domains found in the visited page and in extracted links.
- Resource nodes, representing linked resources loaded by the page.
- Outlier relation, covering foreign/external links and null links.
- Local relation, covering local links associated with the visited domain.
- Relative relation, covering relative links.
- Domain score vector, computed iteratively by the modified authority-ranking algorithm.
- Resource score vector, computed iteratively from domain/resource relations.
- HinPhish score for the target domain, used as the phishing detection signal.

## Methodology

The method crawls a webpage, extracts all relevant links from HTML, maps domains and resource objects into a heterogeneous information network, and models the semantic relationship between the visited page and linked resources. It then iteratively computes domain and resource importance scores. Phishing pages are expected to show less cohesive link relationships, for example many unrelated external or null links copied from a target site. The final HIN-derived feature matrix is used by machine learning classifiers.

## Sources

- Paper page: https://www.mdpi.com/2076-3417/11/20/9733
- DOI: https://doi.org/10.3390/app11209733
