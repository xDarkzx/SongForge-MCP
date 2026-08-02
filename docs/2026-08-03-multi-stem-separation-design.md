# Multi-stem separation (drums/bass/guitar/piano) — Design

Date: 2026-08-03
Status: Approved through discussion. Ready for implementation planning.

## Motivation

Goal (per user, this session): get as close as realistically possible to
Suno's "split into tracks" feature, but free/local — SongForge-MCP
generates a track, this splits it into individual instrument stems, then
reaper-mcp (separate project, untouched by this work) imports them into
a REAPER project as if a real multitrack session existed. That's the
"free instead of paying Suno prices" workflow this unlocks.

Research this session found no dedicated open-source project that does
this end-to-end; the real answer is that `audio-separator` (already
integrated in this codebase for vocals/instrumental splitting) wraps the
same UVR/Demucs model zoo the community already uses as a free
Suno-stems alternative — we just aren't using its non-vocal models yet.

## What's actually available (verified against the installed venv)

Ran `audio-separator --list_models` against the real
`.separator_env` (audio-separator 0.44.5, 167 models) rather than
assuming. Findings that shaped this design:

- Only **one** model in the whole catalog outputs guitar or piano at
  all: `htdemucs_6s.yaml` (Demucs v4, 6-stem: vocals, drums, bass,
  guitar, piano, other).
- It trades accuracy for stem count vs. the 4-stem Demucs models —
  reported SDR: drums 8.5 vs. `htdemucs_ft`'s 10.0, bass 10.1 vs. 12.0.
  Guitar/piano have no reported SDR at all (least-validated outputs in
  the catalog).
- `MDX23C-DrumSep-aufr33-jarredou.ckpt` exists for going *deeper* than a
  single drum stem (kick/snare/toms/hh/ride/crash) — not used in v1, but
  confirms the model zoo goes further if ever needed.
- Confirmed by an actual run (15s clip, `htdemucs_6s`, this session):
  output naming is `{stem}_(Label)_htdemucs_6s.wav` — **the same
  parenthesized-label convention** `separator_client.py` already parses
  for the vocal models. No new parsing scheme needed.

## Decision: single pass, model selectable

User chose `htdemucs_6s` (single pass, all 5 non-vocal stems at once)
over a two-pass higher-accuracy approach (separate `htdemucs_ft` pass
for drums/bass/other + a second `htdemucs_6s` pass just to peel out
guitar/piano) — reasoning: AI-generated source material already has an
audible quality ceiling from the generation step itself, separation adds
further loss on top regardless of model choice, and the gap won't be
worth 2x separation time. EQ/reverb cleanup on the resulting stems is
expected as a normal post-step, not a sign the tool failed.

Both options stay available going forward, at no extra code cost — the
client returns whatever `(Label)` stems the chosen model actually
produces rather than hardcoding an expected stem set, so passing
`htdemucs_ft.yaml` instead just works.

## API design

Extend the existing tool rather than add a new one:

```
split_vocal_stems(audio_path, model=None, extra_stems=None) -> {"job_id": str}
```

- `model`: unchanged — still controls vocal/instrumental separation
  (Roformer default, SDR 12.6).
- `extra_stems` (new, optional): a Demucs model filename, e.g.
  `"htdemucs_6s.yaml"`. When set, also runs that model and returns its
  stems. `None` preserves exactly today's behavior and return shape —
  no breaking change for existing callers.

Job result when `extra_stems` is set gains:

```json
{
  "extra_stems": {
    "drums_path": "...", "bass_path": "...",
    "guitar_path": "...", "piano_path": "...", "other_path": "..."
  },
  "extra_stems_model": "htdemucs_6s.yaml"
}
```

**The Demucs pass runs on the original full mix (`audio_path`), not
chained off the already-separated instrumental.** Demucs is trained on
full mixes with vocals present; feeding it a vocal-stripped file would
be out-of-domain input and could degrade drums/bass/guitar/piano
unpredictably in an untested way. Its own vocals output is discarded —
the dedicated Roformer pass already returns better vocals (SDR 12.6 vs.
Demucs's ~9.6) as `vocals_path`.

## Implementation shape

- `separator_client.py`: generalize the existing 2-bucket
  `_find_stem_output` label-parsing logic (proven against Demucs'
  real output naming this session) into a method that returns *all*
  `(Label)` stems found for a given source/model pair, not just
  vocals/instrumental. Existing `_find_stem_output` / `separate()` stay
  as-is for the vocals/instrumental case — this is additive, not a
  rewrite. New method follows the same idempotency (skip subprocess if
  output already exists for this exact source+model pair), locking, and
  timeout/error-handling pattern as `separate()`.
- `separate_tools.py`: add `extra_stems` param to `split_vocal_stems`,
  call the new client method inside the existing job runner, merge
  results into `job.result`. If the extra-stems pass fails after the
  vocal pass succeeded, the whole job errors out — matches the job
  system's existing all-or-nothing semantics, no new partial-success
  state invented.

## Testing

- Unit tests mocking subprocess, matching `tests/test_separator_client.py`'s
  existing style: correct `extra_stems` model filename passed through,
  correct parsing of Demucs's `(Label)` output into a stems dict, vocals
  excluded from that dict, idempotency (skip subprocess when output
  already exists), a different `extra_stems` model doesn't reuse another
  model's cached output.
- No automated quality/SDR benchmark — there's no ground-truth stem set
  for AI-generated tracks to score against, so that's not computable on
  our own content. Real validation is a single end-to-end run against an
  actual ACE-Step-generated track, listened to by ear.

## Explicitly out of scope for this pass

- `htdemucs_ft` / other model wiring beyond making `extra_stems` accept
  any model filename — no UI/tooling changes to surface a curated list.
- Drum sub-separation (`MDX23C-DrumSep`, kick/snare/toms individually).
- Any reaper-mcp changes — importing the resulting stems into a REAPER
  project is a separate project, untouched here.
