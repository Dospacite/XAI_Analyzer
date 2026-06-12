# Cross-Source Algorithmically Extractable Features

This catalog only includes features that can be extracted from data available in all three MongoDB website sources:

- `phishing_db.website_content`
- `tranco.websites`
- `urlscan.live`

The common usable fields are: requested URL, final URL, title, decoded HTML, HTTP status, and redirect history. For `urlscan.live`, the project normalizes `task.url`, `page.url`, `page.title`, `page.status`, `dom.data`, and redirect responses into that same shape.

Removed from this version:

- URLScan-only fields such as `stats.uniqIPs`, `page.tlsAgeDays`, `page.asn`, `page.country`, and `page.domainAgeDays`.
- RDAP/WHOIS fields, because they are not available for all three sources.
- Response-header fields such as CSP/HSTS/Set-Cookie, because response headers are not consistently available in all three sources.
- Brand-ground-truth features such as known brand names, brand-to-domain mappings, target brand, PageRank, Google index, traffic rank, blacklist membership, and source/label fields.

## URL

| Feature ID | Feature name | Extraction method | Why selected | Paper that uses the feature |
|---|---|---|---|---|
| url.final_url_length | Final URL length | Count characters in `final_url` if present, else `url`. | Phishing URLs are often longer to hide destination or encode tracking/session tokens; also a generic size signal. | Aljofey et al. 2022; Sahingoz et al. 2019; Almomani et al. 2022 |
| url.requested_url_length | Requested URL length | Count characters in requested `url`. | Comparing requested and final URL length can reveal redirect expansion; useful even when not suspicious alone. | Aljofey et al. 2022; Sahingoz et al. 2019 |
| url.scheme_is_https | HTTPS scheme | Parse final URL and test whether scheme is `https`. | HTTPS can be benign hygiene, but phishing also uses HTTPS; useful in combination with other features. | Almomani et al. 2022; Kapan and Gunal 2023 |
| url.hostname_length | Hostname length | Parse final URL hostname and count characters. | Long hostnames can hide the effective domain or contain generated labels. | Sahingoz et al. 2019 |
| url.registrable_domain_length | Registrable domain length | Use TLD extraction and count `domain.suffix`. | Unusually long domains can indicate generated or deceptive domains; generic domain shape signal. | Sahingoz et al. 2019 |
| url.subdomain_length | Subdomain length | Count characters before the registrable domain. | Phishing often uses long subdomains to place trusted-looking words before the real domain. | Sahingoz et al. 2019 |
| url.path_length | Path length | Parse final URL path and count characters. | Long paths may hide payloads, targets, or redirection state. | Sahingoz et al. 2019; Kapan and Gunal 2023 |
| url.query_length | Query length | Parse final URL query and count characters. | Long queries may encode tracking, victim IDs, or relay parameters; generic URL complexity signal. | Kapan and Gunal 2023 |
| url.fragment_length | Fragment length | Parse final URL fragment and count characters. | Fragment content is client-side state and can hide routing or tokens. | Aljofey et al. 2022 |
| url.path_segment_count | Path segment count | Split path on `/` and count non-empty segments. | Deep paths can indicate generated landing pages or cloned directory structures. | Almomani et al. 2022 |
| url.query_parameter_count | Query parameter count | Parse query string and count key/value pairs. | Many parameters can indicate tracking or redirect behavior; generic complexity signal. | Almomani et al. 2022; Kapan and Gunal 2023 |
| url.subdomain_label_count | Subdomain label count | Count labels before the registered domain and public suffix. | Multiple subdomains are common in deceptive hostnames. | Sahingoz et al. 2019; Almomani et al. 2022 |
| url.dot_count | Dot count in URL | Count `.` in the full URL. | Many dots can come from subdomain abuse or embedded hostnames in paths. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.hostname_dot_count | Dot count in hostname | Count `.` in parsed hostname. | Captures subdomain depth more directly than full URL dot count. | Sahingoz et al. 2019 |
| url.hyphen_count | Hyphen count in URL | Count `-` in the full URL. | Hyphens are often used in lookalike or keyword-stuffed domains. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.registrable_domain_hyphen_count | Hyphen count in registrable domain | Count `-` in registered domain only. | Domain-level hyphens are more relevant to deceptive naming than path hyphens. | Almomani et al. 2022 |
| url.underscore_count | Underscore count | Count `_` in final URL. | Extra separators increase URL complexity and can indicate generated URLs. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.slash_count | Slash count | Count `/` in final URL. | Captures path depth and embedded redirection-like structures. | Kapan and Gunal 2023 |
| url.extra_double_slash_count | Double-slash count after scheme | Remove `scheme://`, then count `//`. | Extra `//` can indicate redirect tricks or embedded URLs. | Almomani et al. 2022; Kapan and Gunal 2023 |
| url.question_mark_count | Question-mark count | Count `?` in final URL. | Multiple query separators are unusual and can indicate malformed or obfuscated URLs. | Aljofey et al. 2022 |
| url.equals_sign_count | Equals-sign count | Count `=` in final URL. | Approximates parameter density and encoded state. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.ampersand_count | Ampersand count | Count `&` in final URL. | Captures number of chained query parameters. | Aljofey et al. 2022 |
| url.at_sign_count | At-sign count | Count `@` in final URL. | `@` can obscure the actual host in URLs and is a classic phishing cue. | Aljofey et al. 2022; Almomani et al. 2022 |
| url.percent_sign_count | Percent-sign count | Count `%` in final URL. | High URL encoding can indicate obfuscation. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.hash_sign_count | Hash-sign count | Count `#` in final URL. | Fragment markers can hide routing state or client-side destinations. | Aljofey et al. 2022 |
| url.tilde_count | Tilde count | Count `~` in final URL. | Rare URL symbols are generic lexical anomaly signals. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.plus_sign_count | Plus-sign count | Count `+` in final URL. | Often appears in encoded or generated query text. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.asterisk_count | Asterisk count | Count `*` in final URL. | Rare special character; useful in aggregate special-character profile. | Aljofey et al. 2022; Kapan and Gunal 2023 |
| url.parenthesis_count | Parenthesis count | Count `(` plus `)` in final URL. | Rare URL characters can indicate generated or copied redirect URLs. | Kapan and Gunal 2023 |
| url.square_bracket_count | Square bracket count | Count `[` plus `]` in final URL. | Rare characters and IPv6-style forms are useful URL-shape signals. | Kapan and Gunal 2023 |
| url.curly_bracket_count | Curly bracket count | Count `{` plus `}` in final URL. | Usually uncommon in normal navigational URLs; may indicate templating artifacts. | Kapan and Gunal 2023 |
| url.angle_bracket_count | Angle bracket count | Count `<` plus `>` in final URL. | Malformed or injected-looking URL content is suspicious. | Kapan and Gunal 2023 |
| url.digit_count | Digit count in URL | Count `[0-9]` in final URL. | Generated URLs and fake hostnames often contain many digits. | Sahingoz et al. 2019 |
| url.hostname_digit_count | Digit count in hostname | Count `[0-9]` in parsed hostname. | Numeric hostname labels can indicate generated domains or IP-like disguise. | Sahingoz et al. 2019 |
| url.digit_ratio | Digit ratio in URL | `digit_count / max(1, url_length)`. | Normalizes digit use so short and long URLs are comparable. | Sahingoz et al. 2019 |
| url.special_character_count | Special-character count | Count non-alphanumeric characters in final URL. | Phishing URLs frequently use separators and punctuation for obfuscation. | Sahingoz et al. 2019; Aljofey et al. 2022 |
| url.token_count | URL token count | Split final URL on non-alphanumeric characters and count tokens. | Token count captures URL structural complexity without brand knowledge. | Sahingoz et al. 2019 |
| url.average_token_length | Average URL token length | Average length of URL tokens. | Helps distinguish natural URL words from generated/random segments. | Sahingoz et al. 2019 |
| url.longest_token_length | Longest URL token length | Maximum URL token length. | Very long tokens can indicate random strings, IDs, or encoded payloads. | Sahingoz et al. 2019 |
| url.shortest_token_length | Shortest URL token length | Minimum non-empty URL token length. | Complements average/longest length and helps characterize token distribution. | Sahingoz et al. 2019 |
| url.token_length_stddev | URL token length standard deviation | Standard deviation of URL token lengths. | Captures mixed short separators and long generated tokens. | Sahingoz et al. 2019 |
| url.host_is_ip_address | Host is IP address | Parse hostname and test with `ipaddress.ip_address`. | IP-address hosts hide domain identity and are repeatedly used as phishing cues. | Almomani et al. 2022; Kapan and Gunal 2023 |
| url.punycode_present | Punycode present | Test whether any hostname label starts with `xn--`. | Internationalized-domain encoding can be used for homograph attacks. | Sahingoz et al. 2019 |
| url.explicit_port_present | Explicit port present | Parse URL and test if a port is present. | Non-standard hosting setups can differ between benign and phishing sites. | Almomani et al. 2022 |
| url.non_default_port_present | Non-default port | Port exists and is not 80 for HTTP or 443 for HTTPS. | Non-standard ports can be suspicious or operationally unusual. | Almomani et al. 2022 |
| url.https_token_in_hostname | HTTPS token in hostname | Test whether `https` appears in hostname text. | Attackers may place security words in the domain text to mislead users. | Almomani et al. 2022 |
| url.has_file_extension | URL has file extension | Test whether the final path segment ends with a file-like extension. | File-like paths differ from extensionless application routes and can help characterize phishing kit templates. | Sahingoz et al. 2019 |
| url.path_or_query_contains_url | Path or query contains URL | URL-decode path/query and test for `http://`, `https://`, encoded URL forms, or `www.`. | Embedded URLs in path/query can indicate relay, redirection, or URL-obfuscation behavior. | Almomani et al. 2022; Kapan and Gunal 2023 |
| url.character_entropy | URL character entropy | Compute Shannon entropy over final URL characters. | Randomized URLs tend to have higher entropy; generic anomaly signal. | WebPhish, Opara et al. 2024; Sahingoz et al. 2019 |

## HTML

| Feature ID | Feature name | Extraction method | Why selected | Paper that uses the feature |
|---|---|---|---|---|
| html.length | HTML length | Count characters in decoded HTML. | Raw HTML size is generic context and affects compression/embedding behavior. | HTMLPhish, Opara et al. 2020; WebPhish, Opara et al. 2024 |
| html.visible_text_length | Visible text length | Parse HTML and count visible text excluding script/style/noscript/svg/template. | Text volume distinguishes sparse credential pages from content-rich sites. | Aljofey et al. 2022 |
| html.visible_text_entropy | Visible text entropy | Compute Shannon entropy over normalized visible text. | Very repetitive or highly irregular visible text can indicate generated, boilerplate-heavy, or obfuscated pages; useful with text length and word count. | HTMLPhish, Opara et al. 2020; Aljofey et al. 2022 |
| html.visible_word_count | Visible word count | Tokenize visible text on whitespace and count words. | Generic content-volume feature that complements form/link features. | Aljofey et al. 2022 |
| html.visible_text_to_html_ratio | Visible text to HTML ratio | `visible_text_length / max(1, html_length)`. | Low ratios can indicate script-heavy or image-heavy pages. | Aljofey et al. 2022 |
| html.title_length | Title length | Count characters in extracted title. | Title size captures page identity richness without external ground truth. | Aljofey et al. 2022 |
| html.current_domain_token_in_title | Current domain token in title | Split current domain into local tokens and test if any token appears in title. | Benign pages often self-identify; phishing clones may have inconsistent title/domain context. | Aljofey et al. 2022 |
| html.title_url_token_overlap_ratio | Title/URL token overlap ratio | Tokenize page title and URL hostname/path/query, then divide overlapping tokens by title-token count. | Title and URL mismatch can indicate a page presenting an identity that is not reflected in its actual URL, while overlap can be benign context. | Aljofey et al. 2022 |
| html.title_registered_domain_token_present | Title contains registered-domain token | Test whether the registered domain token appears in normalized title tokens. | Legitimate pages often self-identify with their own domain; absence can support mismatch evidence without brand lists. | Aljofey et al. 2022 |
| html.title_subdomain_token_present | Title contains subdomain token | Test whether any subdomain token appears in the normalized title. | Subdomain/title overlap helps explain whether the visible page identity matches URL components. | Aljofey et al. 2022 |
| html.meta_tag_count | Meta tag count | Count `<meta>` tags. | Metadata density reflects page structure and can combine with redirect/meta-refresh checks. | Almomani et al. 2022 |
| html.meta_refresh_count | Meta refresh count | Count `<meta http-equiv="refresh">`. | Meta refresh is a client-side redirect mechanism used in evasive pages. | Almomani et al. 2022 |
| html.total_tag_count | Total tag count | Count all HTML elements. | Generic DOM complexity signal useful with raw HTML methods. | HTMLPhish, Opara et al. 2020 |
| html.unique_tag_count | Unique tag count | Count distinct HTML tag names. | DOM diversity can separate real sites from simple cloned login pages. | HTMLPhish, Opara et al. 2020 |
| html.anchors_with_href_count | Anchors with href | Count anchors with `href`. | Measures active navigation opportunities. | Aljofey et al. 2022 |
| html.anchors_missing_href_count | Anchors missing href | Count anchors without `href`. | Fake or placeholder links are common in shallow phishing copies. | Aljofey et al. 2022 |
| html.null_or_empty_anchor_count | Null or empty hyperlinks | Count empty, `#`, fragment-only, and JavaScript no-op anchor hrefs. | Placeholder links can imitate a real site without implementing navigation. | Aljofey et al. 2022; Guo et al. 2021 |
| html.placeholder_link_ratio | Placeholder link ratio | Divide null/empty anchor count by all observed anchors. | Many placeholder links suggest a shallow copied page or incomplete navigation. | Aljofey et al. 2022; Guo et al. 2021 |
| html.internal_anchor_count | Internal anchor count | Resolve anchors against final URL and count same registrable domain. | Benign sites often contain many same-domain navigation links. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_anchor_count | External anchor count | Resolve anchors and count different registrable domains. | Copied pages may retain many links to external target resources. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_anchor_ratio | External anchor ratio | `external_anchor_count / max(1, anchors_with_href)`. | Normalized external-link dependence is more comparable across page sizes. | Aljofey et al. 2022 |
| html.external_to_internal_anchor_ratio | External to internal anchor ratio | `external_anchor_count / max(1, internal_anchor_count)`. | High ratios can indicate a page not integrated with its own domain. | Aljofey et al. 2022 |
| html.form_count | Form count | Count `<form>` tags. | Credential collection pages often contain login/payment forms. | Aljofey et al. 2022; Almomani et al. 2022 |
| html.credential_form_present | Credential form present | Test whether a form contains a password input and submit-capable control. | This directly captures credential-collection behavior, one of the clearest phishing-relevant signals. | Aljofey et al. 2022; Almomani et al. 2022 |
| html.password_form_external_action_present | Password form submits off-domain | Test whether any password-containing form has an external action URL. | A credential form posting to another domain is strongly suspicious and easy to explain. | Aljofey et al. 2022; Almomani et al. 2022 |
| html.password_form_null_action_present | Password form has null action | Test whether any password-containing form has missing, empty, fragment, or JavaScript no-op action. | Null credential form actions can indicate copied, deceptive, or script-handled collection flows. | Aljofey et al. 2022 |
| html.null_form_action_count | Null form action count | Count empty, `#`, fragment-only, or JavaScript no-op form actions. | Suspicious form handlers can collect or suppress user input unexpectedly. | Aljofey et al. 2022 |
| html.external_form_action_count | External form action count | Resolve form `action` and count actions to different registrable domains. | Sending credentials off-domain is a strong phishing-relevant behavior. | Aljofey et al. 2022; Almomani et al. 2022 |
| html.mailto_form_action_count | Mailto form action count | Count form actions beginning with `mailto:`. | Email-based form submission is unusual for modern legitimate login flows. | Almomani et al. 2022 |
| html.post_form_count | POST form count | Count forms with `method="post"`. | POST forms are normal for authentication but useful with password/external-action features. | Aljofey et al. 2022 |
| html.input_count | Input count | Count `<input>` tags. | Generic form complexity feature. | Kapan and Gunal 2023; Aljofey et al. 2022 |
| html.text_input_count | Text input count | Count text-like inputs: empty, `text`, `search`, `url`, `tel`, `number`. | Captures data-entry behavior beyond password fields. | Aljofey et al. 2022 |
| html.password_input_count | Password input count | Count `input[type=password]` and password-like name/id/placeholder. | Password fields are central to credential phishing. | Aljofey et al. 2022 |
| html.email_input_count | Email input count | Count `input[type=email]` and email-like name/id/placeholder. | Email fields often appear in account login or recovery flows. | Aljofey et al. 2022 |
| html.hidden_input_count | Hidden input count | Count `input[type=hidden]`. | Hidden fields can carry tokens or destination state; useful with form features. | Almomani et al. 2022 |
| html.hidden_input_ratio | Hidden input ratio | Divide hidden input count by total input count. | Hidden fields can carry tokens or routing state; the ratio normalizes across form sizes. | Almomani et al. 2022 |
| html.submit_button_count | Submit button count | Count `<button type=submit>` and `<input type=submit>`. | Indicates actionable data submission. | Aljofey et al. 2022 |
| html.iframe_count | Iframe count | Count `<iframe>` tags. | Iframes can embed third-party content or hide flows. | Almomani et al. 2022; Kapan and Gunal 2023 |
| html.external_iframe_count | External iframe count | Resolve iframe `src` and count different registrable domains. | Off-domain embedded content can indicate cloaking, delegation, or copied resources. | Almomani et al. 2022 |
| html.script_tag_count | Script tag count | Count `<script>` tags. | Script density helps characterize page behavior and complexity. | Aljofey et al. 2022 |
| html.external_script_count | External script count | Resolve `script[src]` and count different registrable domains. | Off-domain scripts can indicate copied assets or third-party control. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_script_ratio | External script ratio | Divide external script URLs by all same-domain/external script URLs. | Heavy off-domain script dependence can indicate copied assets or externally controlled behavior. | Guo et al. 2021; Aljofey et al. 2022 |
| html.inline_script_count | Inline script count | Count scripts without `src`. | Inline JavaScript can implement redirects or UI manipulation. | HTMLPhish, Opara et al. 2020 |
| html.stylesheet_link_count | Stylesheet link count | Count `<link rel*=stylesheet href>`. | CSS resources are part of cloned visual layout and page structure. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_stylesheet_count | External stylesheet count | Resolve stylesheet href and count different registrable domains. | Copied pages often leave CSS references on external domains. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_stylesheet_ratio | External stylesheet ratio | Divide external stylesheet URLs by all same-domain/external stylesheet URLs. | Hotlinked CSS can indicate copied visual layout or weak domain integration. | Guo et al. 2021; Aljofey et al. 2022 |
| html.image_count | Image count | Count `<img>` tags. | Image density helps distinguish content-heavy sites from minimal forms. | Aljofey et al. 2022 |
| html.external_image_count | External image count | Resolve `img[src]` and count different registrable domains. | Cloned pages may hotlink images from target or CDN domains. | Aljofey et al. 2022; Guo et al. 2021 |
| html.external_image_ratio | External image ratio | Divide external image URLs by all same-domain/external image URLs. | Hotlinked images can reveal cloned visual assets without needing brand knowledge. | Guo et al. 2021; Aljofey et al. 2022 |
| html.favicon_count | Favicon count | Count `<link rel*=icon href>`. | Favicon presence is a page-identity signal. | Almomani et al. 2022 |
| html.external_favicon_count | External favicon count | Resolve favicon href and count different registrable domains. | External favicons can indicate copied branding assets without brand mapping. | Almomani et al. 2022 |
| html.resource_url_count | Resource URL count | Count script, stylesheet, image, favicon, iframe, audio/video/source URLs. | Generic page dependency size; useful with internal/external ratios. | Guo et al. 2021; Aljofey et al. 2022 |
| html.unique_external_resource_domain_count | Unique external resource domain count | Resolve resource URLs and count unique external registrable domains. | A high number of external resource domains can indicate copied, CDN-heavy, or poorly integrated pages. | Guo et al. 2021 |
| html.external_resource_ratio | External resource ratio | External resource URLs divided by all resource URLs. | High off-domain dependency can indicate copied or weakly integrated pages. | Guo et al. 2021; Almomani et al. 2022 |
| html.hidden_element_present | Hidden element indicator | Detect `hidden`, `aria-hidden=true`, hidden inputs, or styles like `display:none`. | Hidden elements may conceal content, traps, or alternate flows. | Almomani et al. 2022 |
| html.javascript_redirect_present | JavaScript redirect indicator | Regex for `location=`, `location.replace`, or `location.assign`. | Client-side redirects are common in evasion and staged landing pages. | Almomani et al. 2022 |
| html.eval_call_count | JavaScript eval call count | Count `eval(` occurrences in HTML/script text. | Dynamic code execution is an obfuscation signal and useful explanation evidence. | HTMLPhish, Opara et al. 2020 |
| html.atob_call_count | JavaScript atob call count | Count `atob(` occurrences in HTML/script text. | Base64 decoding in scripts can indicate obfuscated client-side logic. | HTMLPhish, Opara et al. 2020 |
| html.document_write_count | document.write call count | Count `document.write(` and `document.writeln(` occurrences. | Script-generated page content or redirects are behaviorally relevant and explainable. | HTMLPhish, Opara et al. 2020 |
| html.alert_or_popup_present | Alert or popup indicator | Regex for `alert(` or `window.open(`. | Popups and alerts are interaction behaviors used in some phishing pages. | Almomani et al. 2022 |
| html.onmouseover_handler_count | Onmouseover handler count | Count `onmouseover` attributes or occurrences. | Status-bar manipulation and hover tricks are classic anti-phishing features. | Almomani et al. 2022 |
| html.right_click_disabling_present | Right-click disabling indicator | Regex for `contextmenu`, `event.button == 2`, or `preventDefault()`. | Disabling right-click can obstruct inspection or copying. | Almomani et al. 2022 |
| html.paragraph_count | Paragraph count | Count `<p>` tags. | Generic content-structure feature useful with text and form density. | HTMLPhish, Opara et al. 2020 |
| html.div_count | Div count | Count `<div>` tags. | DOM layout complexity feature. | HTMLPhish, Opara et al. 2020 |
| html.span_count | Span count | Count `<span>` tags. | Fine-grained layout/text structure feature. | HTMLPhish, Opara et al. 2020 |
| html.table_count | Table count | Count `<table>` tags. | Older templates and some phishing kits use table layouts; generic DOM signal. | HTMLPhish, Opara et al. 2020 |
| html.list_count | List count | Count `<ul>`, `<ol>`, and `<dl>`. | Navigation/content richness signal. | HTMLPhish, Opara et al. 2020 |
| html.heading_h1_h2_h3_count | Heading count | Count `<h1>`, `<h2>`, and `<h3>`. | Page structure and text hierarchy help separate sparse forms from complete pages. | Aljofey et al. 2022 |
| html.footer_present | Footer present | Test whether `<footer>` exists. | Full benign sites often include standard layout sections; useful as remediating context. | HTMLPhish, Opara et al. 2020 |
| html.navigation_present | Navigation present | Test whether `<nav>` exists. | Real sites often have navigation; phishing pages may omit it or fake it. | HTMLPhish, Opara et al. 2020 |
| html.privacy_or_terms_link_present | Privacy or terms link present | Search anchor text and href for generic privacy/terms/policy/conditions words. | Complete benign sites often expose policy/terms links; this can be remediating context. | HTMLPhish, Opara et al. 2020 |

## DOMAIN

No `domain.*` features are currently emitted. The previously listed domain features were lexical properties of the hostname or registered domain, so they belong under `url.*` if kept. True domain metadata features, such as RDAP age, registration period, nameserver count, or DNSSEC status, are not available consistently across all three MongoDB sources and are therefore excluded.

## HEADER / HTTP Response

No `header.*` features are currently emitted. Dataset extraction now skips pages whose normalized HTTP status code is not `200`, because status-code features mainly indicate failed fetches, redirects, or crawler artifacts rather than website phishing behavior.

## METADATA

| Feature ID | Feature name | Extraction method | Why selected | Paper that uses the feature |
|---|---|---|---|---|
| metadata.redirect_count | Redirect count | Count normalized redirect history entries. | Phishing flows often use redirection to hide final destinations. | Almomani et al. 2022; Kapan and Gunal 2023 |
| metadata.redirect_domain_change_count | Redirect domain-change count | Walk requested URL, redirect URLs, and final URL; count registrable-domain changes. | Cross-domain movement can indicate staging or off-domain landing. | Guo et al. 2021; Almomani et al. 2022 |
| metadata.final_url_changed | Redirect final URL changed | Compare requested URL to final URL. | Captures whether the fetched page is not the originally requested address. | Almomani et al. 2022 |
| metadata.final_host_changed | Redirect final host changed | Compare requested hostname to final hostname. | Host changes provide stronger redirect context than URL-string changes. | Guo et al. 2021 |
| metadata.final_scheme_changed | Redirect final scheme changed | Compare requested and final URL schemes. | HTTP-to-HTTPS can be normal; HTTPS-to-HTTP can be weakening; useful with other features. | Almomani et al. 2022 |

## Implementation Notes

- Resolve relative links with `urljoin(final_url, href_or_src)`.
- Compare internal and external links by registrable domain, not full hostname.
- Keep label, source collection, Mongo `_id`, screenshot path, and source-specific crawl metadata out of model-visible features.
- Extraction skips non-200 pages and common generic error pages before features are emitted.
- Compression-derived features and exact duplicate/sum features are excluded from the emitted dataset to reduce bloat and shortcut risk.
- Missing values should be explicit, for example `null` plus an availability indicator, rather than imputed from a source-specific default.
