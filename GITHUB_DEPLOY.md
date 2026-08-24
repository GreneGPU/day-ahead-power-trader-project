# GitHub Deployment Guide

This repository is prepared for a public portfolio upload.

## What is included

- Source code under `src/`
- CLI, Streamlit app, strategy simulator, optimizer, and research checks
- Tests and smoke test script
- Reusable `configs/default.json`
- README and project summary
- Empty placeholder folders for `data/`, `outputs/`, and `reports/`

## What is intentionally excluded

- Raw thesis datasets
- Processed/generated data
- Forecast output files
- Generated dashboards and reports
- Local machine config: `configs/jakob-local.json`
- Virtual environment and training logs

## Create a GitHub repository

Create an empty repository on GitHub named, for example:

```text
day-ahead-power-trader-project
```

Do not initialize it with a README, `.gitignore`, or license because this project already has local files.

## Push from PowerShell

From this folder:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial Day-Ahead Power Trading Project"
git remote add origin https://github.com/YOUR_USERNAME/day-ahead-power-trader-project.git
git push -u origin main
```

If Git asks who you are:

```powershell
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

For a public portfolio repo, using your GitHub noreply email is often a good choice.
