# Multi-stem (drums/bass/guitar/piano) separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `split_vocal_stems` so it can also split a mix into non-vocal instrument stems (drums/bass/guitar/piano/other), on top of the existing vocals/instrumental split, using Demucs models already available in the installed `audio-separator` catalog.

**Architecture:** Add a new `extra_stems` optional param to the existing `split_vocal_stems` tool. When set to a Demucs model filename (e.g. `"htdemucs_6s.yaml"`), the job runs a second, independent separation pass on the original full mix (not chained off the already-separated instrumental) and returns whatever non-vocal stems that model produces, keyed as `"{label}_path"`. `None` (the default) preserves today's behavior and return shape exactly.

**Tech Stack:** Python 3.11, `audio-separator` CLI (subprocess), pytest + `monkeypatch` for mocking subprocess calls, existing `songforge_mcp` job-polling pattern (FastMCP tools).

## Global Constraints

- `extra_stems=None` (the default) must not change `split_vocal_stems`'s existing behavior or return shape at all — no breaking change for existing callers.
- The Demucs pass always runs on the original `audio_path` (the full mix), never on the already-separated instrumental file.
- The vocals stem a Demucs model also produces is always discarded from `extra_stems` output — the dedicated vocal-Roformer pass (already returned as `vocals_path`) has meaningfully higher SDR.
- No hardcoded per-model expected stem set — the client returns whatever `(Label)` stems the chosen model's output files actually contain.
- If the `extra_stems` pass fails, the whole job reports an error — no partial-success state, matching the job system's existing all-or-nothing semantics.
- No automated SDR/quality benchmark — there's no ground-truth stem set for AI-generated tracks to score against. Validation is a real end-to-end run, judged by ear.
- Demucs output naming convention (verified this session via a real `htdemucs_6s` run): `{stem}_(Label)_{model_tag}.wav`, e.g. `render_(Guitar)_htdemucs_6s.wav` — same parenthesized-label convention the existing vocals/instrumental code already parses.

---

## Task 1: Client-layer — generalized stem parsing + `separate_extra_stems`

**Files:**
- Modify: `songforge_mcp/separator_client.py` (insert `_find_labeled_stems` after `_find_stem_output`, ~line 73; insert `separate_extra_stems` method after `separate()`, ~line 189)
- Test: `tests/test_separator_client.py`

**Interfaces:**
- Consumes: existing `_STEM_LABEL_RE`, `_model_tag()`, `SeparatorClient._require_configured()`, `SeparatorClient._lock`, `Paths.OUTPUT_DIR`, `Timeouts.SEPARATION`, `ensure_private_dir()`, `no_window_popen_kwargs()`, `ErrorCode`, `SongForgeMCPError` — all already defined in this file/its imports.
- Produces: `_find_labeled_stems(out_dir: str, stem: str) -> dict[str, Path]` and `SeparatorClient.separate_extra_stems(audio_path: str, model_filename: str = "htdemucs_6s.yaml") -> dict[str, str]` (keys like `"drums_path"`, `"bass_path"`, etc. — never includes `"vocals_path"`). Task 2 calls this method by name with these exact signatures.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_separator_client.py` (append at end of file):

```python
def test_find_labeled_stems_parses_demucs_six_stem_output(tmp_path):
    from songforge_mcp.separator_client import _find_labeled_stems

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stem = "mytrack"
    for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
        (out_dir / f"{stem}_({label})_htdemucs_6s.wav").write_text("fake")

    result = _find_labeled_stems(str(out_dir), stem)

    assert set(result.keys()) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    assert result["drums"].name == f"{stem}_(Drums)_htdemucs_6s.wav"


def test_find_labeled_stems_returns_empty_dict_when_no_matches(tmp_path):
    from songforge_mcp.separator_client import _find_labeled_stems

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = _find_labeled_stems(str(out_dir), "mytrack")

    assert result == {}


def test_separate_extra_stems_runs_subprocess_and_excludes_vocals(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert set(result.keys()) == {"drums_path", "bass_path", "guitar_path", "piano_path", "other_path"}
    assert "vocals_path" not in result
    assert result["drums_path"].endswith("(Drums)_htdemucs_6s.wav")


def test_separate_extra_stems_defaults_to_htdemucs_6s(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    client.separate_extra_stems(str(input_path))

    assert "htdemucs_6s.yaml" in captured["cmd"]


def test_separate_extra_stems_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "htdemucs_6s_yaml"
    stems_dir.mkdir(parents=True)
    for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
        (stems_dir / f"render_({label})_htdemucs_6s.wav").write_text(f"fake {label}")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called when extra stems already exist")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fail_if_called)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert set(result.keys()) == {"drums_path", "bass_path", "guitar_path", "piano_path", "other_path"}


def test_separate_extra_stems_reruns_when_existing_stems_are_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    stems_dir = tmp_path / "output" / "stems" / "htdemucs_6s_yaml"
    stems_dir.mkdir(parents=True)
    # Only one stray file present - e.g. a previous run was killed mid-write.
    (stems_dir / "render_(Drums)_htdemucs_6s.wav").write_text("fake drums")

    called = {}

    def fake_run(cmd, **kwargs):
        called["ran"] = True
        out_dir = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out_dir, exist_ok=True)
        for label in ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]:
            with open(os.path.join(out_dir, f"render_({label})_htdemucs_6s.wav"), "w") as f:
                f.write(f"fake {label}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    result = client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")

    assert called.get("ran") is True
    assert len(result) == 5


def test_separate_extra_stems_raises_on_nonzero_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(tmp_path / "output"))
    python_exe = _make_fake_venv(tmp_path)

    input_path = tmp_path / "render.wav"
    input_path.write_text("fake audio")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="model failed to load")

    monkeypatch.setattr("songforge_mcp.separator_client.subprocess.run", fake_run)

    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate_extra_stems(str(input_path), model_filename="htdemucs_6s.yaml")
    assert exc_info.value.code == ErrorCode.SEPARATION_FAILED


def test_separate_extra_stems_raises_when_input_file_missing(tmp_path):
    python_exe = _make_fake_venv(tmp_path)
    client = SeparatorClient(separator_venv_python=python_exe)
    with pytest.raises(SongForgeMCPError) as exc_info:
        client.separate_extra_stems(str(tmp_path / "does_not_exist.wav"))
    assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND
```

These use the file's existing imports/fixtures (`os`, `SimpleNamespace`, `pytest`, `constants`, `SeparatorClient`, `ErrorCode`, `SongForgeMCPError`, `_make_fake_venv`) — no new imports needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_separator_client.py -k "labeled_stems or extra_stems" -v`
Expected: FAIL — `_find_labeled_stems` and `SeparatorClient.separate_extra_stems` don't exist yet (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement `_find_labeled_stems`**

In `songforge_mcp/separator_client.py`, insert immediately after `_find_stem_output` (after its closing `return vocals[0], instrumental[0]` line, before `_SEPARATOR_EXE_NAME = ...`):

```python

def _find_labeled_stems(out_dir: str, stem: str) -> dict[str, Path]:
    """Returns {label.lower(): path} for every {stem}_(Label)_*.wav file
    found in out_dir - e.g. {"drums": ..., "bass": ..., "guitar": ...}.
    Unlike _find_stem_output (which expects exactly two known buckets,
    vocals vs. everything else), this doesn't assume which labels a
    model produces - Demucs models vary (4-stem vs. 6-stem), and the
    caller shouldn't need to hardcode that per model. Returns {} if no
    labeled files are found at all."""
    candidates = list(Path(out_dir).glob(f"{stem}_*.wav"))
    stems: dict[str, Path] = {}
    for f in candidates:
        label_match = _STEM_LABEL_RE.search(f.stem)
        if label_match:
            stems[label_match.group(1).lower()] = f
    return stems

```

- [ ] **Step 4: Implement `separate_extra_stems`**

In `songforge_mcp/separator_client.py`, insert as a new method on `SeparatorClient`, immediately after `separate()`'s closing `return {"vocals_path": str(vocals), "instrumental_path": str(instrumental)}` line, before `def list_models(...)`:

```python

    def separate_extra_stems(self, audio_path: str, model_filename: str = "htdemucs_6s.yaml") -> dict:
        """Returns {"{label}_path": str, ...} for every non-vocal stem
        the given model produces - e.g. htdemucs_6s.yaml (the only
        model in audio-separator's catalog that outputs guitar/piano)
        gives drums_path/bass_path/guitar_path/piano_path/other_path;
        htdemucs_ft.yaml gives drums_path/bass_path/other_path only.
        The vocals stem this model also produces is always dropped -
        call separate() for vocals instead, which uses a dedicated
        model with meaningfully higher vocal SDR.

        Runs on audio_path directly (the original full mix), never on
        an already vocal-separated file - Demucs models are trained on
        full mixes with vocals present, so feeding one a vocal-stripped
        file would be out-of-domain input with unpredictable quality
        impact on the remaining stems.

        Idempotent per (source file, model) pair, same reasoning as
        separate(): if at least two non-vocal stems already exist for
        this exact (audio_path, model_filename) pair, returns those
        without re-running. The "at least two" floor (rather than
        exactly matching the model's full expected stem count, which
        this method deliberately doesn't hardcode) still catches an
        obviously incomplete previous run - e.g. a process killed
        mid-write leaving a single stray file - without needing to know
        in advance how many stems a given model produces."""
        separator_exe = self._require_configured()
        if not os.path.isfile(audio_path):
            raise SongForgeMCPError(
                ErrorCode.FILE_NOT_FOUND, f"audio_path does not exist: {audio_path}"
            )

        model_tag = _model_tag(model_filename)
        out_dir = os.path.join(Paths.OUTPUT_DIR, "stems", model_tag)
        ensure_private_dir(out_dir)
        stem = Path(audio_path).stem

        existing = {k: v for k, v in _find_labeled_stems(out_dir, stem).items() if k != "vocals"}
        if len(existing) >= 2:
            return {f"{label}_path": str(path) for label, path in existing.items()}

        with self._lock:
            try:
                result = subprocess.run(
                    [
                        separator_exe, audio_path,
                        "--output_dir", out_dir,
                        "--output_format", "wav",
                        "--model_filename", model_filename,
                    ],
                    capture_output=True, text=True, timeout=Timeouts.SEPARATION, check=False,
                    **no_window_popen_kwargs(),
                )
            except subprocess.TimeoutExpired as e:
                raise SongForgeMCPError(
                    ErrorCode.SUBPROCESS_TIMEOUT, f"separation exceeded {Timeouts.SEPARATION}s"
                ) from e

        if result.returncode != 0:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"audio-separator exited {result.returncode}: {result.stderr.strip()[-2000:]}",
            )

        stems = {k: v for k, v in _find_labeled_stems(out_dir, stem).items() if k != "vocals"}
        if len(stems) < 2:
            raise SongForgeMCPError(
                ErrorCode.SEPARATION_FAILED,
                f"separation reported success but expected stem output files were not found in {out_dir}",
            )
        return {f"{label}_path": str(path) for label, path in stems.items()}

```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_separator_client.py -v`
Expected: PASS — all tests in the file, including the 8 pre-existing ones and the new ones from Step 1.

- [ ] **Step 6: Commit**

```bash
git add songforge_mcp/separator_client.py tests/test_separator_client.py
git commit -m "Add multi-stem (drums/bass/guitar/piano) separation to SeparatorClient

separate_extra_stems() runs a Demucs-family model (default
htdemucs_6s.yaml) on the original full mix and returns whatever
non-vocal stems it produces, without hardcoding an expected stem set
per model. Generalizes the existing (Label)-parsing logic
(_find_labeled_stems) rather than duplicating it."
```

---

## Task 2: Tool-layer — `extra_stems` param on `split_vocal_stems`

**Files:**
- Modify: `songforge_mcp/tools/separate_tools.py:13-76` (the `split_vocal_stems` tool)
- Test: `tests/test_separate_tools.py`

**Interfaces:**
- Consumes: `SeparatorClient.separate_extra_stems(audio_path: str, model_filename: str = "htdemucs_6s.yaml") -> dict[str, str]` from Task 1, plus existing `_jobs`, `_client`, `validate_output_dir_audio_path`, `Separator.DEFAULT_MODEL`, `SongForgeMCPError`, `ErrorCode` already imported in this file.
- Produces: `split_vocal_stems(audio_path: str, model: str | None = None, extra_stems: str | None = None) -> {"job_id": str}`. On completion, `job.result` gains `"extra_stems": dict[str, str]` and `"extra_stems_model": str` keys only when `extra_stems` was passed — otherwise `job.result` is unchanged from today's shape.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_separate_tools.py` (append at end of file):

```python
def test_split_vocal_stems_with_extra_stems_merges_result(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    def fake_separate_extra_stems(path, model_filename):
        assert model_filename == "htdemucs_6s.yaml"
        return {
            "drums_path": "/fake/drums.wav",
            "bass_path": "/fake/bass.wav",
            "guitar_path": "/fake/guitar.wav",
            "piano_path": "/fake/piano.wav",
            "other_path": "/fake/other.wav",
        }

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)
    monkeypatch.setattr(separate_tools._client, "separate_extra_stems", fake_separate_extra_stems)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path), extra_stems="htdemucs_6s.yaml")
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "complete"
    assert job.result["extra_stems_model"] == "htdemucs_6s.yaml"
    assert job.result["extra_stems"]["drums_path"] == "/fake/drums.wav"
    assert job.result["vocals_path"] == "/fake/vocals.wav"


def test_split_vocal_stems_without_extra_stems_omits_it_from_result(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    def fail_if_called(path, model_filename):
        raise AssertionError("separate_extra_stems must not be called when extra_stems is omitted")

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)
    monkeypatch.setattr(separate_tools._client, "separate_extra_stems", fail_if_called)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path))
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "complete"
    assert "extra_stems" not in job.result


def test_split_vocal_stems_extra_stems_failure_fails_whole_job(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(constants.Paths, "OUTPUT_DIR", str(output_dir))
    audio_path = output_dir / "render.wav"
    _write_tone(audio_path)

    def fake_separate(path, model_filename=None):
        return {"vocals_path": "/fake/vocals.wav", "instrumental_path": "/fake/instrumental.wav"}

    def fake_separate_extra_stems(path, model_filename):
        raise SongForgeMCPError(ErrorCode.SEPARATION_FAILED, "boom")

    monkeypatch.setattr(separate_tools._client, "separate", fake_separate)
    monkeypatch.setattr(separate_tools._client, "separate_extra_stems", fake_separate_extra_stems)

    mcp = _register()
    split_tool = mcp._tool_manager.get_tool("split_vocal_stems")

    async def scenario():
        result = await split_tool.fn(audio_path=str(audio_path), extra_stems="htdemucs_6s.yaml")
        job = separate_tools._jobs.get(result["job_id"])
        for _ in range(50):
            if job.status != "running":
                break
            await asyncio.sleep(0)
        return job

    job = asyncio.run(scenario())
    assert job.status == "error"
    assert "boom" in job.error
    assert job.result is None
```

These use the file's existing imports/fixtures (`constants`, `_write_tone`, `_register`, `separate_tools`, `ErrorCode`, `SongForgeMCPError`) — no new imports needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_separate_tools.py -k "extra_stems" -v`
Expected: FAIL — `split_vocal_stems` doesn't accept `extra_stems` yet (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Implement `extra_stems` param**

In `songforge_mcp/tools/separate_tools.py`, replace the `split_vocal_stems` function (lines 13-76) with:

```python
    @mcp.tool(structured_output=False)
    async def split_vocal_stems(
        audio_path: str, model: str | None = None, extra_stems: str | None = None
    ) -> dict:
        """Start splitting a full mix into a vocals-only stem and an
        instrumental-only stem via audio-separator. Returns {"job_id": str}
        immediately — poll check_vocal_track_status(job_id) exactly as for
        generate_vocal_track (same tool, same registry); on completion it
        returns vocals_path/instrumental_path instead of audio_path.
        Idempotent per (audio_path, model) pair — splitting the same file
        with the same model again returns the existing stems instantly
        instead of re-running; a different model always re-runs.

        Vocal/instrumental separation is never perfect, especially on
        dense/loud mixes where instrumental content masks vocal
        frequencies — some bleed or misclassified passages are possible
        with any model. If results are bad, retry with
        model="model_bs_roformer_ep_368_sdr_12.9628.ckpt" (a different
        architecture/checkpoint) before concluding the audio itself is
        unworkable.

        The default model ("vocals_mel_band_roformer.ckpt") is noticeably
        slower than that alternative on longer files (a 194s run has been
        observed — this is why Timeouts.SEPARATION is sized generously).
        If you need a fast turnaround more than the default's stronger
        vocal isolation, start with
        model="model_bs_roformer_ep_368_sdr_12.9628.ckpt" instead of
        waiting on the default and retrying after a timeout.

        Pass extra_stems (a Demucs model filename, e.g.
        "htdemucs_6s.yaml") to also split the non-vocal instruments out
        of the mix — drums/bass/guitar/piano/other for "htdemucs_6s.yaml"
        (the only model in the catalog that outputs guitar/piano at
        all), or drums/bass/other for "htdemucs_ft.yaml" (higher
        accuracy on those three, no guitar/piano). This runs as a
        second, independent pass on the same original audio_path — not
        chained off the instrumental stem, since Demucs models are
        trained on full mixes with vocals present. On completion the job
        result gains "extra_stems" (a dict of "{label}_path" keys — the
        exact set depends on which model was requested) and
        "extra_stems_model". If this second pass fails, the whole job
        reports an error, even if the vocals/instrumental split itself
        would have succeeded — there is no partial-success result.
        Idempotent the same way as the main split, keyed separately per
        (audio_path, extra_stems model).

        Args:
            audio_path: A file this server previously produced (typically
                generate_vocal_track's audio_path). Must be inside this
                server's own output folder.
            model: audio-separator model filename as a literal string —
                e.g. "vocals_mel_band_roformer.ckpt" or
                "model_bs_roformer_ep_368_sdr_12.9628.ckpt". Call
                list_separator_models() to see the full ranked catalog;
                any "filename" value it returns is valid here. Omit for
                the default. Do NOT pass a Python-style reference like
                "Separator.ALT_MODEL" — that is not a real filename and
                will fail; always use the actual literal filename string.
            extra_stems: Optional Demucs model filename ("htdemucs_6s.yaml"
                or "htdemucs_ft.yaml") to also split out non-vocal
                instrument stems. Omit to get only vocals/instrumental,
                exactly as before.
        """
        audio_path = validate_output_dir_audio_path(audio_path, param_name="audio_path")
        model_filename = model or Separator.DEFAULT_MODEL
        job = _jobs.create()

        async def run_job() -> None:
            try:
                job.message = f"Separating {audio_path} with {model_filename}"
                result = await asyncio.to_thread(_client.separate, audio_path, model_filename)
                job_result = {
                    "vocals_path": result["vocals_path"],
                    "instrumental_path": result["instrumental_path"],
                    "model": model_filename,
                }
                if extra_stems:
                    job.message = f"Separating extra stems with {extra_stems}"
                    extra = await asyncio.to_thread(
                        _client.separate_extra_stems, audio_path, extra_stems
                    )
                    job_result["extra_stems"] = extra
                    job_result["extra_stems_model"] = extra_stems
                job.result = job_result
                job.progress = 1.0
                job.message = "Separation complete"
                job.status = "complete"
            except SongForgeMCPError as e:
                job.error = f"[{e.code.name}] {e.message}"
                job.status = "error"
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        asyncio.create_task(run_job())
        return {"job_id": job.id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_separate_tools.py -v`
Expected: PASS — all tests in the file, including the pre-existing ones and the new ones from Step 1.

- [ ] **Step 5: Commit**

```bash
git add songforge_mcp/tools/separate_tools.py tests/test_separate_tools.py
git commit -m "Wire extra_stems param through split_vocal_stems

Optional extra_stems (a Demucs model filename) runs a second
independent separation pass on the original full mix and merges
drums/bass/guitar/piano/other paths into the job result. Omitting it
preserves today's vocals/instrumental-only behavior exactly."
```

---

## Task 3: Real end-to-end validation + CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`
- No new test file — this task's deliverable is a real separation run against a real track, judged by ear (see Global Constraints: no automated SDR benchmark is possible for AI-generated source material).

**Interfaces:**
- Consumes: the full `split_vocal_stems(audio_path, extra_stems="htdemucs_6s.yaml")` flow from Tasks 1-2, exercised for real (not mocked).

- [ ] **Step 1: Locate or produce a real full-mix test track**

Look in `Vocal-Synth-MCP/output/renders/` (or wherever `Paths.OUTPUT_DIR` resolves for this install) for an existing full mix produced by `generate_vocal_track` — needs actual vocals + instrumentation to be a meaningful test of all 6 stems, not an instrumental-only or vocals-only clip. If none exists, generate a short one first via the acestep skill (`generate_vocal_track` or the ACE-Step UI) — 30-60s is enough for a listening test.

- [ ] **Step 2: Run the real separation, end-to-end**

Write a throwaway script (not committed — delete after use) at the project root, e.g. `scratch_test_stems.py`:

```python
import os
os.environ["SONGFORGE_SEPARATOR_PYTHON"] = os.path.join(
    os.path.dirname(__file__), ".separator_env", "Scripts", "python.exe"
)

from songforge_mcp.separator_client import SeparatorClient

TRACK_PATH = r"PASTE_REAL_FULL_MIX_PATH_HERE.wav"

client = SeparatorClient()
vocals = client.separate(TRACK_PATH)
print("vocals/instrumental:", vocals)

extra = client.separate_extra_stems(TRACK_PATH, model_filename="htdemucs_6s.yaml")
print("extra stems:", extra)
```

Run: `.venv/Scripts/python.exe scratch_test_stems.py`

This will take longer than the earlier 15s test clip did (real track length, GPU permitting — expect low-to-mid tens of seconds per pass on this project's hardware based on the SDR-368 benchmark numbers already observed elsewhere in this codebase). Expected: no exceptions, prints two dicts — `vocals`/`instrumental` paths and 5 `{label}_path` entries (`drums_path`, `bass_path`, `guitar_path`, `piano_path`, `other_path`).

- [ ] **Step 3: Listen to the results**

Open each of the 6 resulting files (vocals, instrumental, drums, bass, guitar, piano, other — note: instrumental and the 5 extra stems will overlap in content, since instrumental is "everything but vocals" from a different model/pass). Report back per-stem: is drums recognizably drums-only, is bass recognizably bass-only, do guitar/piano contain plausible content or mostly bleed/silence (the catalog listed no SDR at all for those two — this is the real, expected first hard data point on whether they're usable). This judgment call belongs to the user, not an automated assertion.

- [ ] **Step 4: Delete the throwaway script**

```bash
rm scratch_test_stems.py
```

- [ ] **Step 5: Add CHANGELOG entry**

Add to the top of the `## Unreleased` section in `CHANGELOG.md`:

```markdown
- **Added multi-stem separation** — `split_vocal_stems` now takes an
  optional `extra_stems` param (a Demucs model filename, e.g.
  `"htdemucs_6s.yaml"`) that runs a second, independent separation pass
  on the original full mix and returns whatever non-vocal stems that
  model produces (`drums_path`/`bass_path`/`guitar_path`/`piano_path`/
  `other_path` for the 6-stem model — the only model in the installed
  catalog that outputs guitar/piano at all, verified via
  `--list_models`; `drums_path`/`bass_path`/`other_path` for the
  higher-accuracy 4-stem `"htdemucs_ft.yaml"`). Runs on the full mix,
  not chained off the vocal-separated instrumental — Demucs models are
  trained on full mixes with vocals present, so a vocal-stripped input
  would be out-of-domain. The model's own vocals output is always
  discarded in favor of the existing dedicated vocal-Roformer split
  (meaningfully higher SDR). Omitting `extra_stems` preserves the
  existing vocals/instrumental-only behavior exactly — no breaking
  change.
```

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md
git commit -m "Document multi-stem separation in CHANGELOG"
```

---

## Self-Review Notes

- **Spec coverage:** design doc's API surface (`extra_stems` param, return shape), full-mix-not-instrumental decision, vocals-discarded decision, all-or-nothing error semantics, idempotency, and "real listening test not SDR benchmark" validation approach are all covered by Tasks 1-3.
- **Type consistency:** `separate_extra_stems(audio_path: str, model_filename: str = "htdemucs_6s.yaml") -> dict[str, str]` is used identically in Task 1's implementation, Task 1's tests, Task 2's tool wiring, and Task 2's tests.
- **Explicitly out of scope** (per design doc, not covered by this plan): drum sub-separation (`MDX23C-DrumSep`), any UI/curated-model-list surfacing beyond accepting any model filename, reaper-mcp changes.
