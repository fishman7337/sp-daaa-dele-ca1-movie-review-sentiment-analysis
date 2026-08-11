# Security Notes

## Repository Checks

CI runs:

- `ruff check .`
- `bandit -r src scripts -ll`
- `pytest`
- scheduled audits of both runtime and notebook dependencies

Dependabot is configured for Python dependencies and GitHub Actions.

## Notebook Runtime

The original notebook contains Colab installation commands and uses external model downloads for augmentation and visualisation. Malay-English augmentation uses Meta's NLLB-200 distilled model locally; it does not send review text to an unofficial translation service. When running it:

- Use a controlled environment.
- Avoid storing credentials in notebooks.
- Do not commit mounted drive paths or private links.
- Review downloaded model licenses before reuse.
- Expect the first translation run to download the NLLB model; cache and checksum model artifacts in controlled environments.

## Secrets

Secrets belong in local environment variables or external secret stores. They must not be committed to Git.
