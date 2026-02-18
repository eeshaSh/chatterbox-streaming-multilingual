import struct
from typing import Optional

import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

model: ChatterboxMultilingualTTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading multilingual model on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("Model loaded.")
    yield


app = FastAPI(title="Chatterbox Multilingual TTS", lifespan=lifespan)

SAMPLE_RATE = 24000
CHANNELS = 1
BITS_PER_SAMPLE = 16


def wav_header(sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS, bits: int = BITS_PER_SAMPLE) -> bytes:
    """WAV header with data size set to max uint32 (unknown length for streaming)."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    # 0xFFFFFFFF signals unknown length
    data_size = 0xFFFFFFFF
    riff_size = data_size  # also unknown
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,            # fmt chunk size
        1,                      # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data", data_size,
    )
    return header


def audio_tensor_to_pcm16(tensor: torch.Tensor) -> bytes:
    """Convert a float audio tensor to int16 PCM bytes."""
    audio = tensor.squeeze().numpy().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


class TTSRequest(BaseModel):
    text: str
    language: str = Field(description="ISO 639-1 language code, e.g. 'en', 'fr', 'zh'")
    audio_prompt_path: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    chunk_size: int = 25


@app.post("/tts")
async def tts_stream(req: TTSRequest):
    def generate():
        yield wav_header()
        for audio_chunk, _metrics in model.generate_stream(
            text=req.text,
            language_id=req.language,
            audio_prompt_path=req.audio_prompt_path,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            chunk_size=req.chunk_size,
            print_metrics=False,
        ):
            yield audio_tensor_to_pcm16(audio_chunk)

    return StreamingResponse(generate(), media_type="audio/wav")


@app.get("/tts")
async def tts_stream_get(
    text: str = Query(...),
    language: str = Query(..., description="ISO 639-1 code"),
    exaggeration: float = Query(0.5),
    cfg_weight: float = Query(0.5),
    temperature: float = Query(0.8),
    chunk_size: int = Query(25),
):
    def generate():
        yield wav_header()
        for audio_chunk, _metrics in model.generate_stream(
            text=text,
            language_id=language,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            chunk_size=chunk_size,
            print_metrics=False,
        ):
            yield audio_tensor_to_pcm16(audio_chunk)

    return StreamingResponse(generate(), media_type="audio/wav")


@app.get("/languages")
async def list_languages():
    return SUPPORTED_LANGUAGES


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
