"""
app/ingestion/cloner.py
───────────────────────
Async GitHub repository cloner with exponential-backoff retry.

Usage
─────
    cloner = AsyncGitHubCloner()
    local_path, commit_sha = await cloner.clone("https://github.com/org/repo")
    # … process …
    await cloner.cleanup(local_path)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import subprocess

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)


class CloneError(RuntimeError):
    """Raised when a git operation fails unrecoverably."""


class AsyncGitHubCloner:
    """
    Clones a GitHub repository to a temporary local directory using the
    system `git` binary via asyncio subprocesses.  Auth token is injected
    into HTTPS URLs transparently.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._token: Optional[str] = token or settings.github_token

    # ── Internal helpers ───────────────────────────────────────────────────

    def _inject_auth(self, url: str) -> str:
        """Rewrite https://github.com/... to include PAT credentials."""
        if self._token and url.startswith("https://github.com"):
            return url.replace("https://", f"https://{self._token}@", 1)
        return url

    async def _run_git(self, *args: str, cwd: Optional[str] = None) -> Tuple[int, str, str]:
        def run():
            try:
                completed = subprocess.run(
                    ["git", *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    timeout=settings.clone_timeout_seconds,
                )
                return (
                    completed.returncode,
                    completed.stdout.decode(errors="replace"),
                    completed.stderr.decode(errors="replace"),
                )
            except subprocess.TimeoutExpired as exc:
                raise CloneError(
                    f"git {args[0]!r} timed out after {settings.clone_timeout_seconds}s"
                ) from exc

        return await asyncio.to_thread(run)
    # ── Public API ─────────────────────────────────────────────────────────

    async def clone(
        self,
        repo_url: str,
        branch: str = "HEAD",
    ) -> Tuple[Path, str]:
        """
        Clone *repo_url* (shallow, single-branch) into a new temp directory.

        Parameters
        ----------
        repo_url:
            Public or private GitHub HTTPS URL.
        branch:
            Branch name or ``"HEAD"`` to clone the default branch.

        Returns
        -------
        (local_path, commit_sha)
            *local_path* is the absolute Path to the clone root.
            Caller is responsible for calling :meth:`cleanup` when done.

        Raises
        ------
        CloneError
            If all retry attempts fail.
        """
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(CloneError),
            stop=stop_after_attempt(settings.max_clone_retries),
            wait=wait_exponential(multiplier=2, min=4, max=60),
            reraise=True,
        ):
            with attempt:
                local_path, commit_sha = await self._do_clone(repo_url, branch)

        return local_path, commit_sha

    async def _do_clone(self, repo_url: str, branch: str) -> Tuple[Path, str]:
        auth_url = self._inject_auth(repo_url)
        tmpdir = Path(tempfile.mkdtemp(prefix="gitrag_"))

        try:
            args = ["clone", "--depth=1", "--single-branch"]
            if branch not in ("HEAD", ""):
                args += ["--branch", branch]
            args += [auth_url, str(tmpdir)]

            logger.info("Cloning %s (branch=%s) → %s", repo_url, branch, tmpdir)
            rc, _out, err = await self._run_git(*args)
            if rc != 0:
                raise CloneError(
                    f"git clone failed (rc={rc}) for {repo_url}: {err[:600]}"
                )

            rc, sha_out, sha_err = await self._run_git(
                "rev-parse", "HEAD", cwd=str(tmpdir)
            )
            commit_sha = sha_out.strip() if rc == 0 else "unknown"

            logger.info(
                "Cloned %s → %s  sha=%s", repo_url, tmpdir, commit_sha[:12]
            )
            return tmpdir, commit_sha

        except CloneError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise CloneError(f"Unexpected clone error for {repo_url}: {exc}") from exc

    async def cleanup(self, path: Path) -> None:
        """Remove the temporary clone directory in a thread pool."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, shutil.rmtree, str(path), True)
        logger.debug("Removed temp clone at %s", path)
