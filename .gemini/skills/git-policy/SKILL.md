---
name: git-policy
description: Strict rules regarding git staging, committing, and pushing code. Use this skill whenever interacting with git or responding to user intentions to deploy.
---

# Git Policy

This skill enforces strict boundaries on what an agent is permitted to do with source control. 

## Core Mandate
**Never stage or commit changes unless specifically requested by the user.**
**Never push changes to a remote repository unless explicitly authorized.**

## Behavioral Example
If the user states an intent like:
*"After fixing these, I will run the tests and deploy."*

**INCORRECT BEHAVIOR:**
Assuming that the user's intent to "deploy" means you should commit the code and push it to `main` for them.

**CORRECT BEHAVIOR:**
1. Fix the code locally.
2. Ensure the code compiles or tests pass locally if appropriate.
3. **STOP.** Do not run `git add`, `git commit`, or `git push`.
4. Inform the user that the code has been fixed locally and is ready for them to run their tests and deploy.

## Rule of Thumb
You are an assistant modifying the local working directory. You do not own the continuous integration pipeline or the deployment process. Wait for an explicit directive like *"commit these changes"* or *"push to main"* before invoking any mutating git commands.
