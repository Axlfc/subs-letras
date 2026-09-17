#!/usr/bin/env python3
"""
subs_letras.py — Genera subtítulos .srt de un vídeo musical (mp4):
separa la voz de la música con Demucs, y luego:

  - Si le das la letra real (.txt): usa FORCED ALIGNMENT (ctc-forced-aligner,
    modelo MMS) para encajar esa letra exacta contra el audio. No hay
    transcripción ni alucinación posible: el texto ya se sabe, solo se
    calcula CUÁNDO se dice cada palabra.

  - Si NO tienes letra: usa whisper.cpp (con DTW + VAD) para transcribir
    Y calcular timings a la vez. Aquí sí puede alucinar en tramos sin voz,
    aunque VAD lo mitiga bastante.

USO
---

  1 argumento — solo el vídeo:
      subs_letras.py video.mp4
      -> genera video.srt en el mismo directorio que el vídeo (vía whisper.cpp)

  2 argumentos:
      subs_letras.py video.mp4 letra.txt
      -> ARG2 es un .txt existente => forced alignment contra esa letra.
         Salida: video.srt (mismo directorio).

      subs_letras.py video.mp4 /ruta/salida/
      subs_letras.py video.mp4 /ruta/salida/nombre.srt
      -> ARG2 NO es un .txt existente => se trata como destino de
         salida (carpeta o fichero .srt). Sin letra -> vía whisper.cpp.

  3 argumentos:
      subs_letras.py video.mp4 letra.txt /ruta/salida(.srt)
      -> ARG2 siempre letra (forced alignment), ARG3 siempre destino.

CONFIGURACIÓN
--------------
Variables de entorno opcionales: WHISPER_BIN, WHISPER_MODEL, DICTADO_LANG,
WHISPER_DTW, WHISPER_VAD_MODEL, DEMUCS_MODEL, ALIGN_LANG_ISO3,
ALIGN_MODEL.
"""

import sys
import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

# --- Configuración whisper.cpp (solo se usa SIN letra) ---
WHISPER_BIN = os.environ.get(
    "WHISPER_BIN", str(Path.home() / "whisper.cpp/build/bin/whisper-cli")
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL", str(Path.home() / "whisper.cpp/models/ggml-large-v3.bin")
)
WHISPER_LANG = os.environ.get("DICTADO_LANG", "es")
WHISPER_DTW_PRESET = os.environ.get("WHISPER_DTW", "large.v3")
WHISPER_VAD_MODEL = os.environ.get(
    "WHISPER_VAD_MODEL", str(Path.home() / "whisper.cpp/models/ggml-silero-v5.1.2.bin")
)

# --- Configuración común ---
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# --- Configuración forced alignment (solo se usa CON letra) ---
# Código ISO 639-3 del idioma de la letra (es -> spa, en -> eng, ca -> cat...)
ALIGN_LANG_ISO3 = os.environ.get("ALIGN_LANG_ISO3", "spa")
ALIGN_MODEL = os.environ.get("ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner")

MAX_LINE_CHARS = 42       # límite de caracteres por línea de subtítulo (norma YouTube ~42)
MAX_LINE_SECONDS = 6.0    # duración máxima de un bloque de subtítulo
GAP_NEW_LINE = 0.6        # si hay más de este silencio entre palabras, se corta línea


# ---------------------------------------------------------------------
# Utilidades de proceso
# ---------------------------------------------------------------------

def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def check_deps(need_whisper: bool):
    missing = []
    for tool in ("ffmpeg", "demucs"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if need_whisper:
        if not Path(WHISPER_BIN).exists():
            missing.append(f"whisper-cli (no encontrado en {WHISPER_BIN})")
        if not Path(WHISPER_MODEL).exists():
            missing.append(f"modelo whisper (no encontrado en {WHISPER_MODEL})")
    if missing:
        print("Faltan dependencias / rutas:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


# ---------------------------------------------------------------------
# Paso 1: extraer audio del vídeo
# ---------------------------------------------------------------------

def extract_audio(video_path: Path, workdir: Path) -> Path:
    audio_path = workdir / "audio.wav"
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", "44100",
        str(audio_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


# ---------------------------------------------------------------------
# Paso 2: separar voz / instrumental con Demucs
# ---------------------------------------------------------------------

def separate_vocals(audio_path: Path, workdir: Path) -> Path:
    print("--> Separando voz e instrumental con Demucs (puede tardar)...")
    run([
        "demucs", "-n", DEMUCS_MODEL, "--two-stems", "vocals",
        "-o", str(workdir), str(audio_path),
    ])
    vocals_path = workdir / DEMUCS_MODEL / audio_path.stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs no generó el fichero esperado: {vocals_path}")
    return vocals_path


def to_whisper_wav(vocals_path: Path, workdir: Path) -> Path:
    out_path = workdir / "vocals_16k.wav"
    run([
        "ffmpeg", "-y", "-i", str(vocals_path),
        "-ac", "1", "-ar", "16000",
        str(out_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


# ---------------------------------------------------------------------
# Vía A (con letra): FORCED ALIGNMENT con ctc-forced-aligner (MMS)
# ---------------------------------------------------------------------

def forced_align(vocals_path: Path, lyrics_text: str):
    """Alinea la letra REAL (ground truth) contra el audio de la voz
    aislada. No hay transcripción ni alucinación: el texto ya se sabe,
    solo se calcula el timing. Devuelve lista de dicts
    {"word": str, "start": float, "end": float}.
    """
    import torch
    from ctc_forced_aligner import (
        load_audio,
        load_alignment_model,
        generate_emissions,
        preprocess_text,
        get_alignments,
        get_spans,
        postprocess_results,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"--> Cargando modelo de forced alignment ({ALIGN_MODEL}) en {device}...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        device, dtype=dtype, model_path=ALIGN_MODEL,
    )

    audio_waveform = load_audio(str(vocals_path), alignment_model.dtype, alignment_model.device)

    print("--> Calculando alineación (puede tardar según duración del audio)...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=8,
    )

    tokens_starred, text_starred = preprocess_text(
        lyrics_text, romanize=True, language=ALIGN_LANG_ISO3,
    )

    segments, scores, blank_token = get_alignments(
		emissions,
		tokens_starred,
		alignment_tokenizer,
	)

    spans = get_spans(tokens_starred, segments, blank_token)

    raw_results = postprocess_results(
		text_starred,
		spans,
		stride,
		scores,
	)

    # El formato exacto de postprocess_results puede variar ligeramente
    # entre versiones del paquete: normalizamos aquí a nuestra forma
    # interna {"word", "start", "end"}.
    words = []
    for item in raw_results:
        if isinstance(item, dict):
            text = item.get("text") or item.get("word")
            start = item.get("start")
            end = item.get("end")
        else:
            # tupla/lista tipo (texto, start, end)
            text, start, end = item[0], item[1], item[2]
        if text and text.strip() and text.strip() != "<star>":
            words.append({"word": text.strip(), "start": float(start), "end": float(end)})

    if not words:
        raise RuntimeError(
            "postprocess_results no devolvió palabras en el formato esperado. "
            "Comprueba la versión instalada de ctc-forced-aligner: "
            "python3 -c \"import ctc_forced_aligner; help(ctc_forced_aligner.postprocess_results)\""
        )

    return words


# ---------------------------------------------------------------------
# Vía B (sin letra): whisper.cpp con timings por palabra
# ---------------------------------------------------------------------

SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\s*\n"
    r"(.*?)(?:\n\n|\Z)",
    re.S,
)


def srt_ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def seconds_to_srt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_words(wav_path: Path, workdir: Path):
    """Ejecuta whisper.cpp con timestamps a nivel de palabra (DTW + VAD)
    y devuelve una lista de dicts {"word": str, "start": float, "end": float}."""
    print("--> Transcribiendo con whisper.cpp (nivel palabra, sin letra de referencia)...")
    out_prefix = workdir / "whisper_words"
    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", str(wav_path),
        "-l", WHISPER_LANG,
        "-ml", "1", "-sow",
        "-osrt",
        "-of", str(out_prefix),
        "-nfa",  # flash-attn rompe DTW, así que la desactivamos siempre
    ]
    if WHISPER_DTW_PRESET:
        cmd += ["-dtw", WHISPER_DTW_PRESET]
    if Path(WHISPER_VAD_MODEL).exists():
        cmd += ["--vad", "-vm", WHISPER_VAD_MODEL]
    else:
        print(f"  (aviso: no se encontró el modelo VAD en {WHISPER_VAD_MODEL} — "
              f"sin VAD, más riesgo de alucinaciones en tramos de silencio)")
    run(cmd)

    srt_file = out_prefix.with_suffix(".srt")
    if not srt_file.exists():
        raise FileNotFoundError(f"whisper.cpp no generó {srt_file}")

    words = []
    content = srt_file.read_text(encoding="utf-8")
    for m in SRT_BLOCK_RE.finditer(content):
        start = srt_ts_to_seconds(m.group(2))
        end = srt_ts_to_seconds(m.group(3))
        text = m.group(4).strip()
        text = re.sub(r"\[.*?\]", "", text).strip()
        if not text:
            continue
        words.append({"word": text, "start": start, "end": end})
    return words


# ---------------------------------------------------------------------
# Agrupar palabras en líneas de subtítulo y escribir el .srt
# ---------------------------------------------------------------------

def group_into_lines(words):
    lines = []
    current = []

    def flush():
        if not current:
            return
        text = " ".join(w["word"] for w in current)
        lines.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": text,
        })

    for w in words:
        if not current:
            current = [w]
            continue
        gap = w["start"] - current[-1]["end"]
        prospective_text = " ".join(x["word"] for x in current + [w])
        duration = w["end"] - current[0]["start"]
        if gap > GAP_NEW_LINE or len(prospective_text) > MAX_LINE_CHARS or duration > MAX_LINE_SECONDS:
            flush()
            current = [w]
        else:
            current.append(w)
    flush()
    return lines


def write_srt(lines, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for idx, line in enumerate(lines, start=1):
            f.write(f"{idx}\n")
            f.write(f"{seconds_to_srt_ts(line['start'])} --> {seconds_to_srt_ts(line['end'])}\n")
            f.write(f"{line['text']}\n\n")


# ---------------------------------------------------------------------
# Resolución de argumentos (1 / 2 / 3 según el patrón pedido)
# ---------------------------------------------------------------------

def resolve_args(argv):
    if len(argv) not in (1, 2, 3):
        print(__doc__)
        sys.exit(1)

    video_path = Path(argv[0]).resolve()
    if not video_path.exists():
        print(f"No existe el vídeo: {video_path}")
        sys.exit(1)

    default_output = video_path.with_suffix(".srt")

    if len(argv) == 1:
        return video_path, None, default_output

    if len(argv) == 2:
        arg2 = Path(argv[1])
        if arg2.suffix.lower() == ".txt" and arg2.exists():
            return video_path, arg2.resolve(), default_output
        output = resolve_output_path(arg2, video_path)
        return video_path, None, output

    lyrics_path = Path(argv[1]).resolve()
    if not lyrics_path.exists():
        print(f"No existe el fichero de letra: {lyrics_path}")
        sys.exit(1)
    output = resolve_output_path(Path(argv[2]), video_path)
    return video_path, lyrics_path, output


def resolve_output_path(raw: Path, video_path: Path) -> Path:
    if raw.suffix.lower() == ".srt":
        raw.parent.mkdir(parents=True, exist_ok=True)
        return raw.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    return (raw / (video_path.stem + ".srt")).resolve()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    video_path, lyrics_path, output_path = resolve_args(sys.argv[1:])
    check_deps(need_whisper=(lyrics_path is None))

    print(f"Vídeo:   {video_path}")
    print(f"Letra:   {lyrics_path if lyrics_path else '(no proporcionada — se usará whisper.cpp)'}")
    print(f"Salida:  {output_path}")

    with tempfile.TemporaryDirectory(prefix="subs_letras_") as tmp:
        workdir = Path(tmp)

        audio_path = extract_audio(video_path, workdir)
        vocals_path = separate_vocals(audio_path, workdir)

        if lyrics_path:
            lyrics_text = lyrics_path.read_text(encoding="utf-8").replace("\n", " ").strip()
            words = forced_align(vocals_path, lyrics_text)
        else:
            vocals_16k = to_whisper_wav(vocals_path, workdir)
            words = transcribe_words(vocals_16k, workdir)

        if not words:
            print("No se ha obtenido ninguna palabra con timing. Revisa el audio separado:")
            print(f"  {vocals_path}")
            sys.exit(1)

        lines = group_into_lines(words)
        write_srt(lines, output_path)

    print(f"\n✅ Subtítulos generados: {output_path}")


if __name__ == "__main__":
    main()
