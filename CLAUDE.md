# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Folder for creation test task.

## Current working instructions
1. First think through the problem, read the codebase for relevant files, and write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

## Coding Conventions MUST FOLLOW!!!

- Compact code without redundant hundreds lines of code is great. No files bigger than 1000 rows.
- Always use relative paths
- All the functions, classes and similar code structures should have a Google format docstrings
- Imports must be organized as in the best python code practices and must be located at the top of the file
- Strict following of SOLID principles and DRY principle.
- Instead of using from Typing import Dict, List... use python built-in typification
- Where possible, the logic should be broke into files, so there are no enourmously large files that have different logic
- Firstly descrive logic in readme, then implement

## Current State

This repository is in its initial state with only a README.md file present. The project structure and implementation files have not yet been created.

## Future Development

When implementing this project, consider:
- Python is likely the intended language

## Project structure
```
├── RAG/        # Old project RAG for example, don't touch
```

## Specialties

