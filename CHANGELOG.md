# Changelog

## Unreleased

- **Fixed a real silent-failure bug: reference-audio uploads could be
  treated as attached when they weren't.** The Playwright automation
  used a blind `wait_for_timeout(6000)` after setting the file input,
  then proceeded regardless — if the upload took longer than 6s,
  generation silently ran on the base model with no reference audio at
  all, indistinguishable from a real voice match in the response. Now
  waits (up to 20s) for the UI to actually confirm the file attached,
  raises a typed error if it never does, and returns
  `reference_audio_confirmed` in the result so `used_reference_audio`
  in tool diagnostics reflects a confirmed attach, not just that a path
  was passed in. Also explicitly propagates `ACESTEP_GENERATION_TIMEOUT`
  to the spawned server process so its own generation watchdog
  (previously defaulting to 600s, unaware of this client's much longer
  `Timeouts.GENERATION` patience) can't kill a legitimately-still-running
  generation out from under it. Reverted `GradioServer.CHECKPOINT`'s
  default from a brief turbo experiment back to `acestep-v15-xl-sft` —
  turbo's distillation bakes in `guidance_scale=1.0` (no CFG), and CFG is
  what makes the model adhere strongly to reference-audio timbre, not
  just text/lyrics; turbo clones reference voices noticeably weaker as a
  direct consequence, confirmed from ACE-Step's own model code.
- **Fixed a real production failure in the new default model
  (`vocals_mel_band_roformer.ckpt`): separation actually succeeded but
  `SeparatorClient` failed to find its own output files.** Two compounding
  bugs, both now fixed with a regression test:
  (1) `_find_stem_output` matched hardcoded `"(Vocals)"`/`"(Instrumental)"`
  literally — the new model writes lowercase `"(vocals)"`/`"(other)"`
  instead, so nothing matched at all.
  (2) The first fix attempt classified by whether `"vocal"` appeared
  anywhere in the filename — still wrong, because this model's own
  filename (`vocals_mel_band_roformer`) is appended as a suffix to
  *every* output file, so both got misclassified as vocals and nothing
  was left for instrumental. Now extracts specifically the parenthesized
  stem label (`"(vocals)"`/`"(other)"`/`"(Instrumental)"`) and classifies
  only that, which is immune to the model's own name containing "vocal".
  Files from affected runs weren't lost — the underlying separation had
  already succeeded — they'll be picked up by the idempotency cache on
  the next call with the same (audio_path, model) instead of re-running.
- **Made `split_vocal_stems`' separator model configurable, and swapped
  the default off audio-separator's own built-in default
  (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`).** Real-world use showed
  vocals bleeding into the instrumental stem badly enough that some vocal
  passages were misclassified as instrumental entirely (not just
  degraded) on dense/loud mixes, where heavy instrumental content masks
  vocal frequencies. New default is `vocals_mel_band_roformer.ckpt` — the
  highest vocal SDR (12.6) of every model in audio-separator's full
  catalog (checked directly via `--list_models`, not assumed), and a
  different architecture from the BS-Roformer model that was actually
  failing, not just a same-family checkpoint swap that likely inherits
  the same weakness. `split_vocal_stems` now takes an optional `model`
  param — pass `Separator.ALT_MODEL` (best-scoring BS-Roformer checkpoint,
  12.1 vocal SDR) to A/B test against the new default on the same source.
  Neither is proven better on this project's actual dense mixes yet —
  this needs a real listening test, not just SDR benchmark numbers, which
  aren't genre-specific. `SeparatorClient.separate()`'s idempotency cache
  is now scoped per (source file, model) via a subfolder per model,
  rather than per source file only — testing a second model against the
  same source no longer silently returns the first model's cached
  result. `install.bat`/`install.sh` now pre-download both candidate
  models during setup instead of a surprise first-use download.
  **Follow-up fix:** the new default model is a different, apparently
  slower architecture than the old one — a real 194s separation was
  observed on a longer file, already past the previous 180s
  `Timeouts.SEPARATION` ceiling (sized around the old model's ~5-15s).
  Raised to 600s with real headroom, same reasoning as `Timeouts.GENERATION`.
  `split_vocal_stems`'s docstring now tells the calling model to reach for
  `Separator.ALT_MODEL` proactively when turnaround matters more than the
  default's stronger vocal isolation, instead of waiting on a timeout and
  retrying reactively.
- **Added `list_separator_models` tool** — returns audio-separator's full
  model catalog (parsed from `--list_models`, sorted by vocal SDR
  descending), so the calling model can discover and suggest any
  separator model, not just the two named `Separator.DEFAULT_MODEL`/
  `ALT_MODEL` constants, when neither is separating a given file cleanly.
  Fast/synchronous — no job polling needed.
- **Fixed a real, confirmed bug class in `_set_field_by_label`: every
  `advanced_settings` write (fill, select, checkbox, radio-fallback) was
  treated as successful whenever the underlying Playwright call didn't
  throw an exception, with zero verification that the value actually
  took effect.** This is the same bug already found once this session in
  a one-off Extract-mode test (a Gradio combobox's `.fill()` typed into
  the filter input without ever registering a real backend selection) -
  turns out the production code path used for every `generate_vocal_track`
  call via `advanced_settings` had the identical gap, never caught before
  because nothing ever read a field back after setting it. Concretely,
  this means every prior "ACE-Step's Key/BPM soft hints are unreliable,
  even when set explicitly" conclusion in this project (including two
  real measured major/minor mode-flips) was reached without ever
  confirming our own automation actually set the field in the first
  place - it's now genuinely unknown which explanation was true until
  this fix ships and a new mismatch (if any) is measured again. Now reads
  every field back after writing it (`.input_value()`/`.is_checked()`)
  and raises `SYNTHESIS_FAILED` if the value didn't actually stick,
  instead of silently reporting success. Numeric comparisons are
  float-tolerant and text comparisons are case-insensitive, since Gradio
  commonly reformats/normalizes values cosmetically without that being a
  real failure. Covered by 8 new tests using a lightweight fake
  Playwright Locator/Page harness.
- **Added instruction guidance for two real, reported quality complaints,
  neither confirmed fixed by a listening test yet:**
  (1) songs losing energy/staying flat across verse→chorus instead of
  building — since ACE-Step has no chord-progression input at all
  (confirmed, grepped the whole codebase) and `caption` is one global
  style description for the entire song, the only available levers are
  describing the dynamic arc explicitly in `caption` prose and annotating
  lyrics structure tags with inline energy/instrumentation cues (e.g.
  `[chorus - heavy, distorted, full energy]` instead of bare `[chorus]`);
  (2) multiple generations in a row feeling same-cadence/"too safe" —
  addressed via three independent `advanced_settings` levers per ACE-Step's
  own documentation: `"Inference Method": "SDE"` (adds real diffusion-
  stage stochasticity, not just seed variation), raised `"LM Temperature"`
  (more creative rhythm/timing choices from the 5Hz LM), and lowered
  `"LM Codes Strength"` (lets the DiT reinterpret the LM's rhythmic plan
  rather than rigidly reproducing it). Explicitly flagged in the
  instructions as unverified-by-listening-test, not a guaranteed fix.

## 0.3.0

- **Set up PyPI publishing** — `songforge-mcp` is now buildable/publishable
  as a real package (added `[project.urls]` and `classifiers` to
  `pyproject.toml`, verified with `twine check`), with a
  `mcp-name: io.github.xDarkzx/songforge-mcp` marker in the README for
  MCP Registry ownership verification. Added
  `.github/workflows/publish.yml`, which builds and publishes to PyPI
  automatically on every GitHub Release via PyPI's trusted-publisher
  (OIDC) flow — no API token stored as a repo secret. Documented the
  one-time PyPI-side trusted-publisher setup and the release process in
  `CONTRIBUTING.md`. `docs/INSTALLATION.md` now mentions `pip install
  songforge-mcp` as an alternative to cloning for the server package
  itself (ACE-Step/separator setup still needs the installer scripts
  either way).
- **Added `SONGFORGE_ACESTEP_CHECKPOINT` env var** so a different ACE-Step
  checkpoint (`acestep-v15-turbo`, `acestep-v15-base`) can be used instead
  of the default `acestep-v15-xl-sft`, previously hardcoded. Deliberately
  a manual, restart-required config option, not a tool — switching
  checkpoints means killing whatever generation is running and a
  multi-GB download, which should never happen automatically or without
  the user's direct say-so.
- **Added 5 new tools** from a full codebase audit's suggestions:
  `generate_vocal_track_takes` (2-5 sequential takes with different
  seeds, each measured via the same BPM/key analysis
  `analyze_reference_audio` uses, so results can actually be compared),
  `list_generated_tracks` and `list_recent_jobs` (browse what's on disk/
  in the job registry when a `job_id` has been lost — a real problem hit
  earlier this project when an orphaned generation had to be manually
  matched to a job ID via log-grepping), `delete_generated_track`, and
  `edit_audio_track` (trim/fade/format-convert). `delete_generated_track`
  deliberately moves files to a `.trash` subfolder rather than actually
  erasing them — raised directly during review: deletion is hard to
  reverse, and a calling model (or a manipulated/injected instruction)
  being wrong about what's safe to remove should never be able to
  permanently destroy a generation. `edit_audio_track`'s mp3 output
  shells out to `ffmpeg` (soundfile can only write wav/flac directly) —
  this project already implicitly depended on `ffmpeg` for YouTube
  reference downloads, now documented as a real requirement in
  `docs/INSTALLATION.md` instead of an unstated one.
- **Security fix: `_extract_video_id` (youtube_reference.py) accepted any
  URL whose text happened to match a YouTube-shaped pattern (e.g.
  `?v=<11 chars>`), with no check that the URL's actual host was
  YouTube.** A URL like `https://evil.example/x?v=aaaaaaaaaaa` extracted
  a "video ID" and passed validation, and the *raw* URL (not a
  reconstructed youtube.com one) was then handed straight to `yt-dlp` —
  which supports 1000+ site extractors with their own history of
  extractor-specific vulnerabilities. Reachable via three public tool
  parameters: `reference_youtube_url`, `remix_source_youtube_url`, and
  `prepare_voice_reference`'s `youtube_url`. Fixed by parsing the URL and
  requiring an exact host match against youtube.com/www.youtube.com/
  m.youtube.com/youtu.be before any ID pattern is even attempted. Found
  via a full security/bug audit of the codebase; everything else checked
  (path traversal, subprocess argument construction, LoRA loading,
  concurrent-generation locking, browser cleanup on exceptions) was
  confirmed already correct, not just assumed clean.
- **Added auto-play for completed generations**, plus a new `play_audio`
  tool for on-demand replay. Real reported problem: a completed
  generation only flashed in the taskbar instead of appearing on screen.
  First attempt launched `wmplayer.exe` (classic Windows Media Player)
  directly with a best-effort foreground-forcing workaround, on the
  theory that the OS default-app path was the cause — **live testing
  then found `wmplayer.exe` creates zero visible windows at all on
  Windows 11** (confirmed via a full window enumeration by PID); Windows
  11 replaced it with a modern "Media Player" app that plain
  `os.startfile()` already opens correctly and visibly. Reverted to the
  simple default-app approach (`open_with_default_app`) before shipping
  the wmplayer-specific version, which would have been a regression. New
  `play_audio(audio_path)` tool covers the case where auto-play still
  doesn't work or the user wants to replay an earlier result.
- **Correction to the entry below, from watching a real timeout happen
  live via ACE-Step's own log:** raising this server's own polling
  ceiling to 1800s does not fix this failure mode on its own. The
  actual error was `TimeoutError: Music generation timed out after 600
  seconds` raised from *inside ACE-Step's own code*
  (`generate_music_execute.py`) — a hardcoded internal watchdog
  entirely separate from this server's `Timeouts.GENERATION`, which it
  fires well before regardless of what that's set to. Real numbers from
  the live log: `batch=2, steps=50, duration=251.0s` — `Batch Size` had
  been left at ACE-Step's own default of 2 (generating two full songs
  at once, roughly doubling compute) and duration was confirmed unset
  (`"original was None/unset"` in the log), so ACE-Step's own CoT step
  picked 251s from the long lyrics. Per-step diffusion time was ~7.5x
  slower than a comparable successful generation as a result. Added an
  instruction requiring both `"Audio Duration (seconds)"` (~180-240)
  *and* `"Batch Size": 1` to be set explicitly on every call unless the
  user asks otherwise — this is the real fix; the timeout increase
  below is still reasonable to keep as a generous outer ceiling, but
  was never sufficient by itself.
- **Diagnosed a real "Timed out — GPU ran out of steam" complaint** on a
  full-length song with long lyrics. Not a bug: this project's GPU sits
  in the 12-16GB tier ACE-Step's own `GPU_COMPATIBILITY.md` documents as
  "marginal" for the XL DiT model this server uses, requiring real CPU
  offload — that tier still permits requesting up to 8-minute songs, and
  long lyrics add further LM/conditioning work on top of an already
  offload-slowed tier. The generation timeout was a flat 900s regardless
  of requested duration (confirmed in code — no scaling logic existed at
  all), and was only ever validated against short/typical clips.
  Raised to 1800s and documented the real reasoning. Paired with a new
  instruction defaulting `"Audio Duration (seconds)"` to ~3-4 minutes
  unless the user explicitly asks for longer, and requiring a heads-up
  before honoring a longer request, so long waits aren't mistaken for a
  hang and short requests aren't needlessly slow by defaulting to "Auto".
- **Diagnosed a real "it just stopped" complaint on a live `split_vocal_stems`
  call via Claude Desktop's own MCP server logs, not speculation.** The
  logs (`mcp-server-songforge.log`) showed the calling model genuinely
  started the split and polled `check_vocal_track_status` 6 times over
  ~2 minutes, each call getting a valid server response — then simply
  stopped calling it. No stem files were ever produced on disk for that
  track, confirming the job was abandoned mid-poll (most likely still
  legitimately running, given separation's own subprocess timeout is
  180s and only ~120s of polling had elapsed), not a server-side crash
  or silent failure. Root cause: this project's own polling instructions
  said to "poll" and "stay quiet while polling" without explicitly
  stating this must be a loop that continues until a terminal
  `"complete"`/`"error"` status — worded ambiguously enough that a
  single poll followed by silence was a plausible (mis)reading.
  Rewrote both `generate_vocal_track`'s and `split_vocal_stems`'
  instructions to state explicitly: call `check_vocal_track_status`
  again and again until a terminal status, a `"running"` result is
  never a stopping point, and ending a turn without a final outcome
  message is a failure even when the underlying job succeeded.
- **Diagnosed a real "wrong mood" complaint on a live generation, not a
  hypothetical.** A track generated for a deliberately dark, minor-key
  concept ("Hollow") came back sounding wrong. Measured with this
  server's own `analyze_audio`: BPM 161.5 against a ~150 half-time
  target, and — the real culprit — key landed on **C# major** against
  an intended D/F **minor**. A full major/minor flip, not an adjacent-key
  miss. Ruled out corruption first (no clipping, no dropouts, normal
  peak/RMS) before concluding it was a musical mismatch, not a broken
  render. Root cause: ACE-Step's `Key`/`BPM` fields are documented
  elsewhere in this project as soft hints, not hard constraints, but
  this is the first *measured* case of a full modality flip rather than
  a nearby-key miss. Added an instruction requiring explicit
  `advanced_settings` for BPM/key whenever a specific target was agreed
  on (not just mood language in the caption), plus a mandatory
  post-generation `analyze_reference_audio` check against the intended
  target before presenting a result as done.

- **Added and exhaustively tested `remix_no_fsq`, closing out the Remix
  mode investigation.** ACE-Step's "no_fsq" bypasses FSQ (finite scalar
  quantization) of the source's structure in favor of continuous latents.
  It raised objective tonal-clarity scores, which looked like a real fix
  — but a real listening test showed why that metric was misleading: the
  result was a near-exact reproduction of the source track, new lyrics
  not used at all. This revealed the actual shape of Remix mode's
  limitation on this checkpoint: every real setting swept (strength,
  melody retention, fsq quantization) trades one failure (garbled vocals
  that don't resemble the source) for the opposite one (a clean-sounding
  but literal copy that ignores the new lyrics) — not a matter of finding
  the right combination. Documented plainly in the tool's own docstrings
  and injected instructions so this doesn't need re-discovering.
- **Fixed a real cross-process race condition that was silently corrupting
  generation quality:** `ensure_server_running`'s "is it up, launch if not"
  check only guarded against duplicate launches within one process
  (`asyncio.Lock`) — it did nothing to stop a second OS process (e.g. this
  MCP server's own startup warm-up racing with a separately-run script)
  from also concluding "not running yet" and launching a second ACE-Step
  server. Confirmed happening for real: two `acestep_v15_pipeline.py`
  processes were both running, both had loaded the ~9GB XL-SFT model onto
  the same 12GB GPU (95% VRAM utilization), and the next real generation
  timed out after 600s. Added a cross-process advisory lock (atomic
  exclusive file creation, with stale-lock reclamation if a holder
  crashed) so only one process ever launches the server; others wait for
  it. Covered by new tests.
- **Added `remix_melody_retention` to `generate_vocal_track`**, exposing
  ACE-Step's "Cover Strength (Melody Retention)" as an experimental knob
  for Remix mode's gibberish/garbled-vocal problem. Tested at 0.0
  (ACE-Step's own documented "pure style transfer" value) — confirmed by
  a real listening test to make things worse, not better ("music jumping
  and glitching," no longer resembling the source either). Documented as
  not a fix; left at ACE-Step's own default unless a specific reason to
  deviate is confirmed. Combined with the Reference Audio trim fix and
  Remix mode's already-confirmed problems, this closes out a full round
  of testing: every audio-conditioned generation approach tried (Reference
  Audio, Remix at multiple strength/retention combinations) measured and
  sounded clearly worse than caption-only generation, which remains the
  one reliable path.
- **Corrected a documentation claim that turned out to be wrong when
  actually verified against ACE-Step's source.** Every place in this
  project (tool docstrings, injected instructions, docs/TOOLS.md,
  docs/ARCHITECTURE.md, youtube_reference.py) claimed reference audio
  controls "vocal timbre only, never content" — copied from ACE-Step's
  own UI description without checking the actual mechanism. Tracing
  `infer_refer_latent()` in ACE-Step's
  `acestep/core/generation/handler/conditioning_embed.py` shows reference
  audio is conditioned on via a real acoustic VAE latent of the reference
  clip (same encoder used for real audio), not a narrow timbre-only
  signal — so it plausibly affects instrumentation/production texture
  too, not just voice character. Corrected everywhere, with an honest
  flag that exact perceptual strength vs. `caption` hasn't been confirmed
  by an actual listening test yet. Also added a missing workflow step:
  the injected instructions never actually told the calling model to
  attach a user-provided reference (local file or YouTube link) to the
  `generate_vocal_track` call — only to describe reference audio's
  semantics if used. That gap plausibly explains a real report of the
  model verbally acknowledging a YouTube reference link ("great, I'll use
  that as reference") without ever actually passing it through — the
  server log showed zero trace of `reference_youtube_url` ever being
  used, confirming the tool itself was never actually invoked with it.
- **Fixed a real bug hit right after the `output_format` change went
  live:** a completed generation failed with "could not read rendered
  audio... unknown format: 3" even though the file existed and was fine.
  `measure_wav_duration_seconds` used Python's stdlib `wave` module,
  which only understands canonical integer-PCM WAV and cannot read
  32-bit float PCM (format code 3) — which is what a real "WAV (16-bit)"
  ACE-Step generation was observed actually landing as on disk. Switched
  to `soundfile` (libsndfile-backed, already a dependency), which reads
  float/extensible WAV correctly; verified directly against the exact
  file that had failed (confirmed `subtype: FLOAT`, now reads its real
  225.0s duration). Added a regression test covering this exact shape.
- Added MCP server startup warm-up: `main.py` now kicks off ACE-Step's
  cold-start as soon as the MCP server itself launches (i.e. as soon as
  Claude Desktop starts it), via the same `ACEStepClient` instance/lock
  `generate_vocal_track` uses (a separate instance would risk two
  competing server launches racing for the same port), so the checkpoint
  load happens in the background before the user even asks for a track
  instead of blocking their first request.
- **Added `output_format` to `generate_vocal_track`**, defaulting to
  uncompressed `"wav"` instead of ACE-Step's own `"mp3"` UI default —
  better for importing into a DAW for further editing. Also supports
  `"flac"`, `"opus"`, `"aac"`, `"wav32"`. Required driving a previously
  unused part of ACE-Step's UI (a global, localStorage-persisted Settings
  panel preference, not a per-generation form field, and not a native
  `<select>`) — confirmed correct end-to-end with a real generation that
  produced an actual `.wav` file on disk, not just verified at the
  UI-interaction level. See `docs/ARCHITECTURE.md` for the full story.
- **Moved the default output location out of the OS temp directory** and
  into an `output/` folder inside this repo checkout (override with
  `SONGFORGE_OUTPUT_DIR`). Temp turned out to be a real problem: system
  cleanup tools can purge it, backups typically skip it, and it isn't
  somewhere a user would think to look for a track they want to keep —
  found when a completed track had to be located via server logs instead
  of just being where you'd expect. The default deliberately stays
  inside this repo rather than pointing at any sibling project folder,
  so a fresh standalone clone works the same way for anyone.
- Added a `note` field to completed `generate_vocal_track`/`split_vocal_stems`
  responses pointing at the real `audio_path`/stem paths on disk, as a
  fallback for clients that don't render inline MCP `AudioContent` yet.
  Found via a real case: Claude Desktop showed "unsupported format" for a
  generation that had actually succeeded — confirmed via the server log
  (no error, 2 content blocks returned) and the output file itself
  (valid, playable, correct duration) — the failure was in the client's
  rendering of the audio content block, not this server or the track.
- **Fixed a real timeout bug found via live testing against Claude
  Desktop:** a generation call was cancelled by the client after a fixed
  240s with zero progress notifications ever delivered (MCP's
  `report_progress` is a silent no-op without a client-supplied
  `progressToken`, which Claude Desktop doesn't send). `generate_vocal_track`
  now starts generation as a background job and returns a `job_id`
  immediately; new `check_vocal_track_status(job_id, wait_seconds=25.0)`
  tool long-polls it (blocks briefly server-side while still running, so
  the calling model doesn't need its own delay between polls).
- **Walked back over-eager narration** added alongside the above fix: an
  initial version told the calling model to post a chat update on every
  single poll, which per direct user feedback produced "way too much
  feedback... spamming" and burned message/credit budget for no new
  information. Instructions now call for a handful of messages per
  generation at most (start, maybe one mid-way check-in, then done/error)
  — the job/poll split was the actual fix for the timeout; narration
  volume was a separate, over-corrected concern.
- `split_vocal_stems` is now called only when the user explicitly asks
  for the vocal/instrumental split out — not automatically after every
  generation.
- Fixed console windows flashing/appearing during `yt-dlp` and
  `audio-separator` subprocess calls on Windows (`capture_output=True`
  alone doesn't suppress this) via a new shared
  `no_window_popen_kwargs()` helper, applied to every subprocess call in
  this codebase.
- Documented recommended system specs (GPU/VRAM, disk, Python, OS) in
  `docs/INSTALLATION.md` and `README.md`, sourced from ACE-Step 1.5's own
  published requirements rather than estimated.
- Both tools now return generated/separated audio inline as playable MCP
  `AudioContent`, not just a file path string.
- `split_vocal_stems`' `audio_path` is now restricted to this server's
  own output folder (`Paths.OUTPUT_DIR`) — it can no longer be pointed at
  an arbitrary file elsewhere on disk. `reference_audio_path` gained
  real validation (must exist, must have an audio extension, must parse
  as genuine audio content via `soundfile`) without that folder
  restriction, since pointing at a personal sample library anywhere on
  disk is its legitimate use.
- Cross-platform support: added `install.sh` (Linux/macOS) alongside
  `install.bat`. Replaced the PowerShell `Start-Process`-based detached
  server launch in `acestep_client.py` with native Python `subprocess`
  detachment flags (also removes an f-string shell-injection surface).
  `.venv` layout and the separator executable name are now OS-detected
  rather than hardcoded to Windows paths.

## 0.2.0

- Full rewrite around ACE-Step 1.5 (Playwright-driven Gradio UI
  automation) replacing the original DiffSinger-based tool contract
  entirely. New tools: `generate_vocal_track`, `split_vocal_stems`.
  DiffSinger's note/phoneme-based approach is not carried forward in any
  form — see `docs/ARCHITECTURE.md` for why.

## 0.1.0

- Initial v1: `synthesize_vocal`, `list_voicebanks`, `validate_score` MCP
  tools. Subprocess integration with a separately-cloned openvpi/DiffSinger
  checkout. LUNAI Project's "Katyusha" voicebank as the configured default.
