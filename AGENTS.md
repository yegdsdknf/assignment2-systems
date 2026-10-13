# AI Agent Guidelines for CS336 at Stanford

This file provides instructions for AI coding assistants (like ChatGPT, Claude Code, GitHub Copilot, Cursor, etc.) working with students in CS336.


# Development environment

This project is stored in and executed inside WSL Ubuntu.

## Command execution

All Python, Git, testing, package-management, and build commands must run inside WSL.

When operating from the Windows-native Codex agent, run commands using:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /home/lty/assignment1-basics && <command>"
```

Replace `<command>` with the actual command.

Examples:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /home/lty/assignment1-basics && git status"
wsl.exe -d Ubuntu -- bash -lc "cd /home/lty/assignment1-basics && python3 main.py"
wsl.exe -d Ubuntu -- bash -lc "cd /home/lty/assignment1-basics && pytest"
```

Do not use Windows-native Python, Git, Node.js, package managers, virtual environments, or build tools for this project.

## File handling

* Preserve Linux-style LF line endings.
* Preserve executable permissions on shell scripts.
* Do not edit virtual environments, Conda environments, `node_modules`, caches, or generated files unless explicitly requested.
* Use Linux paths when running commands inside WSL.
* Read and edit source files through the opened WSL project directory.

## Verification

After modifying code:

1. Run the relevant tests inside WSL.
2. Run formatting or linting tools when configured.
3. Review the changes with `git diff`.
4. Report any test or command that could not be completed.
