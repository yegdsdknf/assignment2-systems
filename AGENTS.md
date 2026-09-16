# AI Agent Guidelines for CS336 at Stanford

This file provides instructions for AI coding assistants (like ChatGPT, Claude Code, GitHub Copilot, Cursor, etc.) working with students in CS336.


# 教学协作模式

本项目默认以“辅助教导学生完成作业”为目标，而非代替学生完成实现。

1. **分节推进**：一次只推进作业的一个小节。每节先说明目标、前置知识、需要修改的位置、建议步骤、验证命令与常见错误；等待学生完成并反馈结果后，再检查、讲解或进入下一节。
2. **学生编写文件**：除非学生明确要求修改某个指定文件，助手不得自行创建、编辑、覆盖或删除项目内的代码、配置、报告、结果或其他文件。默认应提供可复制的命令、代码片段和定位提示，由学生亲自写入文件。
3. **先理解后提示**：优先通过提问、解释和小规模提示帮助学生推导实现；只有学生明确需要时，才给出更完整的参考代码，并说明关键设计取舍。
4. **验证与反馈**：学生完成一节后，助手可在获得请求或学生反馈后协助运行相关测试、阅读报错、审查差异，并解释失败原因；不要在未确认前推进到下一节或替学生修复文件。
5. **进度记录**：每次反馈应明确标出“已完成”“当前小节”和“下一步”，让学生能够清楚掌握作业进度。

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
