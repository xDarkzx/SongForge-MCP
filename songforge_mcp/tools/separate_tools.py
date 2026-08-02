import asyncio

from mcp.server.fastmcp import FastMCP

from songforge_mcp.shared_state import jobs as _jobs, separator_client as _client
from songforge_mcp_shared.constants import Separator
from songforge_mcp_shared.error_codes import ErrorCode, SongForgeMCPError
from songforge_mcp_shared.protocol import validate_output_dir_audio_path


def register(mcp: FastMCP):
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

    @mcp.tool(structured_output=False)
    async def list_separator_models(vocal_only: bool = True) -> dict:
        """List audio-separator's available vocal-separation models,
        sorted by vocal SDR score descending. Use this to pick or suggest
        a specific model for split_vocal_stems's `model` param when the
        default ("vocals_mel_band_roformer.ckpt") or its usual alternative
        ("model_bs_roformer_ep_368_sdr_12.9628.ckpt") aren't separating
        cleanly — pass any "filename" value from this list's results
        directly to split_vocal_stems's `model` param as a literal
        string. Fast, synchronous, no job polling needed (typically ~1s,
        no GPU/audio processing involved).

        Args:
            vocal_only: Exclude models with no vocal-stem score at all
                (denoise/deverb/crowd-removal models etc. that don't do a
                vocal/instrumental split). Default True.
        """
        models = await asyncio.to_thread(_client.list_models, vocal_only)
        return {"models": models}
