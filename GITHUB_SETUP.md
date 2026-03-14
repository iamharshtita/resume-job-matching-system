# GitHub Repository Setup Guide

## Step 1: Initialize Local Git Repository

```bash
# Navigate to project directory
cd "/Users/harshtita/Desktop/ASU Docs/Data Mining- Spring26/Course Project/resume-job-matching-system"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Project structure and design plan

- Set up multi-agent resume-job matching system structure
- Added base agents (Resume Parser, Skill Miner, Orchestrator)
- Created configuration and requirements files
- Added comprehensive design plan
- Included data directory structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

## Step 2: Create GitHub Repository

### Option A: Using GitHub CLI (Recommended)

```bash
# Login to GitHub (if not already logged in)
gh auth login

# Create repository
gh repo create resume-job-matching-system \
  --public \
  --description "Multi-agent AI system for intelligent resume-job matching with explainable results" \
  --source=. \
  --remote=origin \
  --push
```

### Option B: Using GitHub Web Interface

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `resume-job-matching-system`
   - **Description**: `Multi-agent AI system for intelligent resume-job matching with explainable results`
   - **Visibility**: Public (or Private for your team)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

4. Connect local repo to GitHub:
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/resume-job-matching-system.git

# Push code
git branch -M main
git push -u origin main
```

## Step 3: Set Up GitHub Project Board (Optional but Recommended)

1. Go to your repository on GitHub
2. Click "Projects" tab
3. Click "New project"
4. Choose "Board" template
5. Name it "Resume-Job Matching Development"
6. Create columns:
   - **Backlog**: Not yet started
   - **In Progress**: Currently working on
   - **Review**: Awaiting code review
   - **Done**: Completed tasks

## Step 4: Add Collaborators

1. Go to repository Settings
2. Click "Collaborators" (left sidebar)
3. Click "Add people"
4. Enter teammate usernames/emails
5. Set permissions (Write or Admin)

## Step 5: Set Up Branch Protection (Recommended)

1. Go to Settings → Branches
2. Click "Add rule"
3. Branch name pattern: `main`
4. Enable:
   - ✅ Require pull request before merging
   - ✅ Require approvals (1 reviewer)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
5. Save changes

## Step 6: Create Development Workflow

### Branching Strategy

```bash
# For new features
git checkout -b feature/resume-parser
# ... make changes ...
git add .
git commit -m "Implement resume parser with spaCy"
git push origin feature/resume-parser

# Create PR on GitHub
gh pr create --title "Add Resume Parser Agent" --body "Implements resume parsing with name, education, and experience extraction"

# After PR approved and merged
git checkout main
git pull origin main
git branch -d feature/resume-parser
```

### Branch Naming Convention
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `refactor/component-name` - Code refactoring
- `docs/update-description` - Documentation
- `test/test-description` - Test additions

## Step 7: Add GitHub Actions (Optional)

Create `.github/workflows/tests.yml` for automated testing:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    - name: Run tests
      run: pytest tests/
```

## Quick Reference Commands

```bash
# Check status
git status

# Create and switch to new branch
git checkout -b feature/new-feature

# Stage changes
git add file.py
# or stage all
git add .

# Commit with message
git commit -m "Your message"

# Push to GitHub
git push origin branch-name

# Pull latest changes
git pull origin main

# View branches
git branch -a

# Delete local branch
git branch -d branch-name

# View commit history
git log --oneline --graph

# Create PR via CLI
gh pr create

# View PRs
gh pr list

# Checkout PR locally
gh pr checkout PR_NUMBER
```

## Troubleshooting

### Authentication Issues
```bash
# For HTTPS (recommended)
gh auth login

# For SSH
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add to GitHub: Settings → SSH and GPG keys
```

### Large Files Warning
If you get warnings about large files:
```bash
# Add to .gitignore
echo "data/raw/*.pdf" >> .gitignore
git rm --cached data/raw/*.pdf
git commit -m "Remove large files from tracking"
```

### Merge Conflicts
```bash
# Pull latest changes
git pull origin main

# Fix conflicts in files (look for <<<<<<, ======, >>>>>>)
# After fixing:
git add fixed_file.py
git commit -m "Resolve merge conflict"
git push
```

## Best Practices

1. **Commit Often**: Small, focused commits are better than large ones
2. **Write Clear Messages**: Describe what and why, not how
3. **Review Before Pushing**: Always check `git diff` before committing
4. **Pull Before Push**: Always pull latest changes before starting work
5. **Never Commit Secrets**: Use .env files (already in .gitignore)
6. **Test Locally**: Run tests before pushing
7. **Use PRs**: Don't push directly to main
8. **Review Others' Code**: Help catch bugs early

## Repository URL Format

After setup, your repository will be at:
```
https://github.com/YOUR_USERNAME/resume-job-matching-system
```

Share this with your team!
