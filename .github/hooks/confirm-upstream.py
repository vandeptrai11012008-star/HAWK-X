import json
import re
import sys


INSTALL_OR_CLONE = re.compile(
    r"""\bgit\s+clone\b
    |\bpython\s+-m\s+pip\s+install\b
    |\bpip(?:3)?\s+install\b
    |\buv\s+(?:pip\s+)?install\b
    |\bpoetry\s+add\b
    |\b(?:npm|pnpm|yarn)\s+(?:install|add)\b
    |\bcargo\s+install\b
    |\byolo(?:\.exe)?\s+(?:predict|train|val|export)\b
    |\bpython\s+-m\s+ultralytics\b""",
    re.IGNORECASE | re.VERBOSE,
)


def main():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return

    arguments = event.get("toolArgs", event.get("tool_input", {}))
    if isinstance(arguments, str):
        command = arguments
    elif isinstance(arguments, dict):
        command = arguments.get("command", "")
    else:
        command = ""

    if not isinstance(command, str) or not INSTALL_OR_CLONE.search(command):
        print("{}")
        return

    print(json.dumps({
        "permissionDecision": "ask",
        "permissionDecisionReason": (
            "Confirm the source, target environment, and run scope before "
            "cloning a repository, installing packages, or running Ultralytics."
        ),
    }))


if __name__ == "__main__":
    main()
