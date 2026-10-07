#!/usr/bin/env bash
# Save a bot's state and redraw the README's progress block, then push.
#
#   bash scripts/commit_state.sh "📊 swing EOD 2026-10-06 [skip ci]"
#
# The last step of every bot's workflow, run from the repo root. The three
# bots push to the same branch, and the progress block (README.md and
# docs/progress/, drawn by scripts/progress.py) is built from all of their
# state, so two bots finishing together would collide on it. So the commit
# goes through `git pull --rebase` holding state/ alone (each bot writes only
# its own state/<mode>/, which never conflicts), and the block is redrawn on
# top of whatever the others pushed, just before each push. A push that loses
# a race drops its redraw and starts again. Failing to draw never stops the
# state from being saved.

set -u
branch="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
pause="${RETRY_PAUSE:-5}"
git config user.name  "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git add state/
if git diff --staged --quiet; then
  echo "state unchanged"
  exit 0
fi
git commit -q -m "$1"

for i in 1 2 3 4; do
  if git pull -q --rebase origin "$branch"; then
    state_only=$(git rev-parse HEAD)
    if python3 scripts/progress.py; then
      git add README.md
      [ -d docs/progress ] && git add docs/progress
      git diff --staged --quiet || git commit -q --amend --no-edit
    else
      echo "::warning::could not redraw the progress block; saving state without it"
    fi
    git push -q origin "HEAD:$branch" && exit 0
    git reset -q --hard "$state_only"
  else
    git rebase --abort 2>/dev/null
  fi
  sleep $((i * pause))
done
echo "::error::could not push state after 4 attempts"
exit 1
