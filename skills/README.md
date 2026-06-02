# multivon-eval Claude Code skills

Three Claude Code skills that integrate `multivon-eval` into your
Claude Code workflow. Drop into `~/.claude/skills/` and Claude Code
auto-discovers them.

## Available skills

| Skill | What it does | When it auto-invokes |
|---|---|---|
| `eval-bootstrap` | Generates a runnable eval suite from your product description + sample traces. Bootstrap CLI wrapped in a Claude Code workflow. | "add evals", "set up evaluation", or detected LLM imports without an eval/ directory. |
| `eval-audit` | Pre-flight eval check on a PR diff. Runs only the cases that stress the changed surface. Blocks safety-class regressions. | After `/review`, on diffs that touch prompts, model calls, tool definitions. |
| `eval-explain` | Explains why a particular evaluator was recommended, in 3 sentences. | After `/eval-bootstrap`, or on user phrases like "why did multivon pick X". |

## Install

The skills ship in the `multivon-eval` PyPI package under
`multivon_eval/skills/`. To make them available to Claude Code, symlink
them into your skills directory:

```bash
# After: pip install multivon-eval
PKG_PATH=$(python -c "import multivon_eval, pathlib; print(pathlib.Path(multivon_eval.__file__).parent)")
mkdir -p ~/.claude/skills
ln -sf "$PKG_PATH/../skills/eval-bootstrap" ~/.claude/skills/eval-bootstrap
ln -sf "$PKG_PATH/../skills/eval-audit"     ~/.claude/skills/eval-audit
ln -sf "$PKG_PATH/../skills/eval-explain"   ~/.claude/skills/eval-explain
```

Or for the convenience-CLI path (recommended once shipped):

```bash
multivon-eval install-skills          # writes the symlinks for you
```

Verify Claude Code sees them:

```bash
ls ~/.claude/skills/
# Expect: eval-audit  eval-bootstrap  eval-explain  (plus any others you have)
```

## Why ship skills alongside the framework

Anthropic's skill model lets CLI tools teach Claude Code how to use
them correctly. Without a skill, Claude Code has to infer evaluator
selection / threshold calibration / bootstrap flow from your docs — and
hallucinates command names half the time. With a skill, the tool's own
team writes the workflow once and every Claude Code session inherits
it. The Quarkdown / Anthropic skill-creator / gstack ecosystems are all
converging on this pattern.

If you want to extend or fork these skills, the SKILL.md spec
([reference](https://docs.anthropic.com/claude/skills)) is plain
Markdown with YAML frontmatter — no DSL to learn.
