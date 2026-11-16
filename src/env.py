"""Optional loading of a local, git-ignored .env file."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_env(path: str | Path = PROJECT_ROOT / ".env") -> bool:
    """Load variables from the project's .env file if python-dotenv is installed.

    Existing environment variables are never overridden, and no values are logged.
    Returns True when a file was found and loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = Path(path)
    if not env_path.is_file():
        return False
    return load_dotenv(env_path, override=False)
