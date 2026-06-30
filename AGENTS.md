# Repo Command Guidance

- For user-facing commands, never prefix commands with inline environment variables like `STABLEWM_HOME=... python ...` or `MPLCONFIGDIR=... python ...`.
- If an environment value is needed, write it to `.env` or pass the value as a normal program argument.
- Prefer commands that load repo environment from `.env`:
  ```bash
  set -a
  source .env
  set +a
  ```
- Inline environment variables are acceptable only for quick internal verification, never in final user-facing commands.
