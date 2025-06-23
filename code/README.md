Based on the content from the ModelScope community website, here's the English translation and explanation:

# Do a website!

This repository is designed for complex code generation work, primarily targeting two types of projects: frontend React projects and native JavaScript projects.

The codebase contains three YAML configuration files:

- **workflow.yaml** - The entry configuration file for code generation; the command line automatically detects this file's existence
- **agent.yaml** - Configuration file used for generating React projects, referenced by workflow.yaml
- **native.yaml** - Configuration file for native JS projects; can be used by manually modifying the `config:` field in workflow.yaml to run support for native JS projects

This project needs to be used together with ins-agent.

## Running Commands

First install ins-agent:
```shell
pip install ins -U
```

Then run:
```shell
# use with native.yaml
ins run --config ins/coding --modelscope_api_key xxx --trust_remote_code true
# use with agent.yaml
ins run --config ins/coding --openai_api_key xxx --trust_remote_code true
```

The modelscope_api_key can be obtained from the [https://www.modelscope.cn/my/myaccesstoken](https://www.modelscope.cn/my/myaccesstoken) page.

After the command starts, simply describe a requirement to generate code. The code will be output to the "output" folder in the current directory by default.

## Architecture Principles

The workflow is defined in workflow.yaml and follows a two-phase approach:

**Design & Coding Phase:**
1. A user query is given to the architecture
2. The architecture produces PRD (Product Requirements Document) & module design
3. Optional: An architecture reviewer does one round of review, and the architecture reproduces the PRD and module design
4. The architecture starts several programmer tasks to finish the coding jobs
5. The Design & Coding phase completes when all coding jobs are done

**Refine Phase:**
1. The first three messages are carried to the refine phase (system, query, and architecture design)
2. Building begins (in this case, npm install & npm run build); error messages are incorporated into the process
3. The refiner distributes tasks to programmers to read files and collect information (these tasks do no coding)
4. The refiner creates a fix plan with the information collected from the tasks
5. The refiner distributes tasks to fix the problems
6. After all problems are resolved (or exceed 20 problems), users can input additional requirements, and the refiner will analyze and update the code accordingly

This system appears to be an automated code generation and refinement tool that can create complete web applications based on natural language descriptions, with built-in error detection and fixing capabilities.