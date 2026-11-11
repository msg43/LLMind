# LLMind

This repo has no GAMEPLAN.md or CHANGELOG.md yet; create them when the first real task finishes.

# GAMEPLAN.md is the cold-start file

A new session should get the project's direction from `GAMEPLAN.md` for a few thousand tokens, instead of carrying the history in a long conversation. That only works if the file is short, keeps everything that still shapes future work, and is updated on a schedule.

**What goes in it** (about 200 lines in all, roughly 6K tokens)
- The goal, and the current state in a few lines.
- **Strategy and constraints:** the approach and why it was chosen, the rules that must not be broken, and the known pitfalls. This section is required.
- What is in progress, the next three steps, and decisions still open (each with its owner).
- Decisions taken, each with its reason and date, so a restarted session does not re-argue them.
- Where to look for detail (code map, changelog, archive sections by heading and line).

**Keep or archive test:** if something still shapes future work (a strategy, a constraint, a decision's reason, an unresolved item, a pitfall that could recur), it stays in `GAMEPLAN.md`. Only finished work, evidence and superseded material move to `CHANGELOG.md` or `docs/archive/`, and the opening section links to it. When in doubt, keep the one-line version and link the detail.

**When to update it**
- When a task finishes, a decision is made, or a blocker appears, in the same change.
- Before a session ends, and before a restart or compaction. Write the handoff first: what is done, what is not, and what waits on Matthew.
- When the context advisory fires (400K tokens), as a checkpoint.
- Resolved items leave in the same update, so the file does not grow.

**Size check:** if the file is past about 200 lines or 30K tokens, archive per the test above before doing anything else. Each time you archive, read what you removed once and confirm nothing that still shapes future work went with it.

# Keep sessions cheap to start and to run

Every turn re-reads the whole conversation, so a session that starts small and stays small costs far less.

- **Start from `GAMEPLAN.md`.** Read it first; its opening section is written to be enough to begin work. Read only the top of `CHANGELOG.md` (newest entries come first, so about the first 100 lines). Use grep to find older entries. Never read a whole changelog or an archive file; read the line range you need.
- **Changelog entries are short.** About five lines each: what changed, why, and where to look. Detail belongs in the commit message or the pull request. The changelog is append-only history and is never trimmed or archived: its size costs nothing as long as sessions read only the top and grep for the rest. Use `grep -m` or a few lines of context so a search returns a short result.
- **Grep before reading.** Search for the name, then read only the lines around it. Send a large file, log or batch of sessions to a small helper agent that returns a short answer, as the helper-agent rule describes.
- **Restart before the context gets heavy.** At about 400K tokens, update `GAMEPLAN.md` with the handoff (done, not done, waiting on Matthew) and continue in a fresh session, as your global instructions describe.

# Recovering something deleted from GAMEPLAN.md

Git keeps every version of `GAMEPLAN.md`, so do not save dated copies. To find a strategy, constraint, decision or to-do that went missing:

- Find when a phrase left the file: `git log -S"some phrase" --oneline -- GAMEPLAN.md`. The newest commit listed is usually the one that removed it.
- Read the file as it was then: `git show <commit>:GAMEPLAN.md`, or see exactly what a commit removed: `git show <commit> -- GAMEPLAN.md`.
- Sweep for anything lost: list the lines each commit removed with `git log -p --format="== %h %ad %s" --date=short -- GAMEPLAN.md`, then check each removed line against the keep-or-archive test in this file and restore what is still live. Sweep by time window (for example one month per helper agent) and write findings to a file, as the helper-agent rule describes.

