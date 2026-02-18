import io
import struct
import torch
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from chatterbox.tts import ChatterboxTTS
from typing import Optional
import tempfile
import os

app = FastAPI()

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading model on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)
print("Model loaded.")

SAMPLE_RATE = model.sr  # 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def make_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int) -> bytes:
    """Create a WAV header with unknown data size (streamed)."""
    # Use 0xFFFFFFFF for data size to indicate unknown/streaming
    data_size = 0xFFFFFFFF
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size,  # will be wrong but that's fine for streaming
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def audio_stream(
    text: str,
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    chunk_size: int = 25,
    output_format: str = "pcm",
):
    """Generator that yields raw PCM (or WAV with header) bytes from generate_stream."""
    if output_format == "wav":
        yield make_wav_header(SAMPLE_RATE, NUM_CHANNELS, SAMPLE_WIDTH * 8)

    for audio_chunk, metrics in model.generate_stream(
        text=text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        chunk_size=chunk_size,
        print_metrics=True,
    ):
        # audio_chunk: torch.Tensor of shape (1, num_samples), float32 in [-1, 1]
        audio_np = audio_chunk.squeeze().cpu().numpy()
        # Convert float32 -> int16 PCM
        pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
        yield pcm_data


@app.get("/tts")
async def tts(
    text: str = Query(..., description="Text to synthesize"),
    exaggeration: float = Query(0.5, description="Emotion exaggeration factor"),
    cfg_weight: float = Query(0.5, description="Classifier-free guidance weight"),
    temperature: float = Query(0.8, description="Sampling temperature"),
    chunk_size: int = Query(25, description="Tokens per chunk"),
    format: str = Query("pcm", description="Output format: 'pcm' (raw int16) or 'wav'"),
):
    """Stream TTS audio as raw PCM or WAV."""
    content_type = "audio/wav" if format == "wav" else "audio/L16;rate=24000;channels=1"

    return StreamingResponse(
        audio_stream(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            chunk_size=chunk_size,
            output_format=format,
        ),
        media_type=content_type,
    )


@app.post("/tts")
async def tts_post(
    text: str = Form(..., description="Text to synthesize"),
    audio_prompt: Optional[UploadFile] = File(None, description="Reference audio for voice cloning"),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
    chunk_size: int = Form(25),
    format: str = Form("pcm"),
):
    """Stream TTS audio with optional voice cloning via file upload."""
    audio_prompt_path = None
    tmp_file = None

    if audio_prompt is not None:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.write(await audio_prompt.read())
        tmp_file.close()
        audio_prompt_path = tmp_file.name

    def stream_and_cleanup():
        try:
            yield from audio_stream(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size,
                output_format=format,
            )
        finally:
            if tmp_file is not None:
                os.unlink(tmp_file.name)

    content_type = "audio/wav" if format == "wav" else "audio/L16;rate=24000;channels=1"

    return StreamingResponse(
        stream_and_cleanup(),
        media_type=content_type,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
