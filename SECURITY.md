<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 John Luke NIKABOU (LucNIK) -->

# Security and privacy policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/LucNIK/valia/security/advisories/new),
not in a public issue. You will get an answer within 72 hours.

## Personal data (DVF)

DVF is published under the Licence Ouverte 2.0 with two conditions that this project enforces:

- **No re-identification.** The application never shows the exact address of a sale: comparable sales
  are shown with a rounded position (about 100 m) and the month only. No owner data exists in DVF and
  none is ever added.
- **No search-engine indexing.** There is no page per sale or per address. Sales are only displayed in
  the browser after a user asks for an estimate, in a block marked `data-nosnippet`, and `robots.txt`
  disallows crawling of the data files (`/data/`).

The application sets no cookie and collects no personal data. Estimates are computed in the browser;
only the typed address and its position are sent, to the national address service (IGN Géoplateforme),
to find the address and its cadastral parcel. The theme choice
is remembered in the browser's local storage. A strict Content Security Policy only allows the site
itself and the address service.
