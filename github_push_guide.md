# GitHub Push Guide: Secure Deployment Planner

As the Architect and Security Advisor, I have prepared this step-by-step terminal execution guide. Follow these instructions carefully in your VS Code terminal to safely initialize, verify, and push your repository to GitHub without leaking sensitive credentials.

> [!CAUTION]
> **NEVER** push your `.env` file to a public repository. If an API key is leaked, it can be scraped by bots within seconds, potentially costing you money or compromising your accounts. Always verify your staging area before committing.

---

### Step 1: Security Audit
Before initializing Git, visually confirm that your `.gitignore` is present and explicitly lists the files we want to keep secret. Run this command to verify `.env` is covered:

```bash
# Windows PowerShell command to check for .env in .gitignore
Select-String -Path .gitignore -Pattern "\.env"
```
*You should see `.env` listed in the output. If you do not, **STOP** and add `.env` to your `.gitignore`.*

### Step 2: Local Initialization
Now that security is verified, initialize your local Git repository:

```bash
git init
```

### Step 3: Staging & Verification (CRITICAL)
This is the most important step. We will check the status **before** and **after** staging to ensure no credentials slip through.

First, check what Git sees:
```bash
git status
```
*Look at the "Untracked files" list. You should **NOT** see `.env`, `__pycache__/`, or `data/chroma_db/` here.*

If it looks clean, stage all files:
```bash
git add .
```

Verify the staging area one last time:
```bash
git status
```
*Review the "Changes to be committed" list. Ensure `.env` is absolutely **NOT** on this list.*

### Step 4: Professional Committing
Create a standardized, professional initial commit message that reflects the production-ready state of your project:

```bash
git commit -m "feat: initial commit of UMKM RAG Assistant with Hybrid Retrieval and Groq/Gemini routing"
```

### Step 5: Remote Configuration
1. Open your browser and log into [GitHub](https://github.com/).
2. Click the **"+"** icon in the top right and select **"New repository"**.
3. Name your repository (e.g., `umkm-rag-assistant`).
4. **DO NOT** check "Add a README file" or "Add .gitignore" (we already created these locally).
5. Click **"Create repository"**.

Copy the remote URL provided by GitHub, and link it to your local repository using the following command (replace the URL with your actual GitHub URL):

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Step 6: Final Push
Rename your default branch to `main` (the modern standard) and securely push your code to GitHub:

```bash
git branch -M main
git push -u origin main
```

---
> [!SUCCESS]
> **Congratulations!** Your project is now securely hosted on GitHub and ready to be showcased in your portfolio.
