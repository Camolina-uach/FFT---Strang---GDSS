# GitHub pre-publication checklist

Complete these items before making the repository public:

- [x] Publication under BSD-3-Clause is authorized by the copyright holders
  (confirmed 2026-08-17).
- [ ] The author names and order in `CITATION.cff` are confirmed.
- [x] The repository is created under the agreed account.
- [x] The repository URL is added to `CITATION.cff` and both READMEs.
- [ ] The unit tests and reduced workflow pass in GitHub Actions.
- [ ] A release named `v1.0.0` is created from the exact article version.
- [ ] The GitHub repository is connected to Zenodo and the release is archived.
- [ ] The Zenodo DOI is added to `CITATION.cff` and the code-availability statement.
- [ ] The manuscript points to the archived release rather than an unversioned branch.

Do not add generated `outputs/` to Git. The small reference CSV tables are the
version-controlled regression data; full derived outputs belong in the DOI-backed
archive if they are required by the journal or reviewers.
