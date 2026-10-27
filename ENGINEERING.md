# Engineering rules

Every rule here exists because it was broken once, in this repo, and cost something real.
None of them are general advice. Where a rule can be enforced by a test, the test is named.

## Resources

**Measure a reservation before you make one.** ZeroGPU charges the
`@spaces.GPU(duration=...)` reservation, not the time used. A run that takes 60s was asking
for 150s, burning 2.5× the free daily allowance per call for nothing. Time the work, then
reserve just above it.

**Spend the user's quota on their behalf, not on your testing.** Hosted free tiers are
per-account and shared with whatever the user is doing. A Gemini daily quota was exhausted
by verification runs, so a real narration failed later that day. Batch verification into as
few calls as possible, cache results, and prefer a cached artefact over a fresh call.

**Default to the cheap path and make the expensive one opt-in.** Every shot gets depth
parallax, which is free and local. Image-to-video is a slider that starts at zero. A
feature that silently spends quota on every run is a bug.

**Never let a paid or quota-limited step fail the whole job.** An animated clip that does
not arrive leaves the still in place. A narration that fails leaves the film silent. The
render always completes.

## Dependencies

**Check pins before installing anything.** Installing `gradio_client` upgraded
`huggingface-hub` 0.36 → 2.3, and `transformers` pins `<1.0`. Nothing failed at install
time; CLIP, depth estimation and narration simply stopped loading, and an unrelated check
happened to notice. Three `requests` calls replaced the package.

> Enforced by `test_installed_packages_have_consistent_requirements` and
> `test_the_local_model_stack_still_imports`.

**A dependency that breaks the local pipeline is not worth the convenience it buys.** Write
the HTTP calls.

## The user's data

**Never modify a project you were asked to diagnose.** "Test 2" was re-analysed and
re-rendered in place, which overwrote its title, its settings, its shot plan and its video.
The before/after comparison the user wanted survived only as numbers. Build a new project.

**Read what you are about to destroy.** `git add -A` staged 206 files of copyrighted manga
for a public repo, caught only by a pre-push audit. `.gitignore` now excludes archives and
common scan folders, but the habit is the real guard.

**Tests no longer share the developer's database — keep it that way.** They used to, and
it cost three separate incidents: a routing fixture called `reset()` with no argument and
deleted real measurements; a label used in a test but missing from the fixture's
allow-list was never cleaned up and stayed in the live table as a fake endpoint; and the
suite quietly accumulated 237 projects in the real database.

`tests/conftest.py` now sets `KAGE_DB` to a temp file before the first `import backend`,
which is the only moment it can work — `backend.config` reads it while importing, and
`backend.database` builds its engine from the result. `test_the_test_run_is_not_using_the_real_database`
asserts the isolation holds, because a regression here is otherwise silent.

Scoping cleanup to the rows a test created is still good manners, but it is a discipline
that has to be remembered every single time. A separate file has to be remembered once.

## Claims

**Measure, then state the number.** Several comments in this repo cite measurements — depth
VRAM, provider latency, how often a camera move left the frame blank. If you change the
behaviour behind one, re-measure and update it. A stale number is worse than none, because
the next reader trusts it.

**Verify a regression test fails without the fix.** Two "regression tests" here passed with
their fix reverted: one because a numpy array never reached the code path, one because
`TestClient` pools a single connection. Both are now labelled as smoke tests with the
limitation written in the docstring. A test that cannot fail is documentation wearing a
test's clothes.

**Check the vendor's own docs, not a blog.** Blog posts said Hugging Face Spaces had a free
Docker tier; HF's own docs say Gradio and Docker Spaces require a paid plan. Blogs said
Colab tolerated tunnels; Google's FAQ names web-UI content generation as a common cause of
free-tier termination. Both would have been wrong recommendations.

## Secrets

**`.env` is the only place keys live, and it is git-ignored.** Keys are never logged, never
returned by an API, and never printed — the providers endpoint is covered by
`test_provider_endpoint_never_returns_keys`.

**Tell the user when a key has been exposed.** Anything pasted into a chat or a terminal
transcript should be rotated, and saying so is part of the job.
