#!/usr/bin/env python3
"""
subs_letras.py — Genera subtítulos .srt de un vídeo musical (mp4):
separa la voz de la música con Demucs, transcribe con whisper.cpp
(timings palabra a palabra), y opcionalmente corrige alucinaciones
comparando contra la letra real.

USO
---

  1 argumento — solo el vídeo:
      subs_letras.py video.mp4
      -> genera video.srt en el mismo directorio que el vídeo

  2 argumentos:
      subs_letras.py video.mp4 letra.txt
      -> ARG2 es un .txt existente => se usa como letra real para
         corregir alucinaciones. Salida: video.srt (mismo directorio).

      subs_letras.py video.mp4 /ruta/salida/
      subs_letras.py video.mp4 /ruta/salida/nombre.srt
      -> ARG2 NO es un .txt existente => se trata como destino de
         salida (carpeta o fichero .srt). Sin corrección de letra.

  3 argumentos:
      subs_letras.py video.mp4 letra.txt /ruta/salida(.srt)
      -> ARG2 siempre letra, ARG3 siempre destino de salida.

CONFIGURACIÓN
--------------
Ajusta las rutas de whisper.cpp en las constantes de más abajo, o
exporta las variables de entorno WHISPER_BIN / WHISPER_MODEL antes de
ejecutar el script.
"""

import sys
import os
import re
import subprocess
import shutil
import tempfile
import difflib
from pathlib import Path

# --- Configuración (puedes sobreescribir con variables de entorno) ---
WHISPER_BIN = os.environ.get(
    "WHISPER_BIN", str(Path.home() / "whisper.cpp/build/bin/whisper-cli")
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL", str(Path.home() / "whisper.cpp/models/ggml-large-v3-turbo.bin")
)
WHISPER_LANG = os.environ.get("DICTADO_LANG", "es")
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")  # separación voz/instrumental

MAX_LINE_CHARS = 42       # límite de caracteres por línea de subtítulo (norma YouTube ~42)
MAX_LINE_SECONDS = 6.0    # duración máxima de un bloque de subtítulo
GAP_NEW_LINE = 0.6        # si hay más de este silencio entre palabras, se corta línea


# ---------------------------------------------------------------------
# Utilidades de proceso
# ---------------------------------------------------------------------

def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def check_deps():
    missing = []
    for tool in ("ffmpeg", "demucs"):
        if shutil.which(tool) is None:
            missing.append(tool)
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
    # Demucs crea: workdir/<modelo>/<nombre_audio_sin_ext>/vocals.wav
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
# Paso 3: transcripción palabra a palabra con whisper.cpp
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
    """Ejecuta whisper.cpp con timestamps a nivel de palabra y devuelve
    una lista de dicts: {"word": str, "start": float, "end": float}"""
    print("--> Transcribiendo con whisper.cpp (nivel palabra)...")
    out_prefix = workdir / "whisper_words"
    run([
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", str(wav_path),
        "-l", WHISPER_LANG,
        "-ml", "1", "-sow",
        "-osrt",
        "-of", str(out_prefix),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    srt_file = out_prefix.with_suffix(".srt")
    if not srt_file.exists():
        raise FileNotFoundError(f"whisper.cpp no generó {srt_file}")

    words = []
    content = srt_file.read_text(encoding="utf-8")
    for m in SRT_BLOCK_RE.finditer(content):
        start = srt_ts_to_seconds(m.group(2))
        end = srt_ts_to_seconds(m.group(3))
        text = m.group(4).strip()
        # whisper.cpp puede meter marcadores tipo [SPEAKER_TURN] o dejar
        # bloques vacíos en silencios; los descartamos
        text = re.sub(r"\[.*?\]", "", text).strip()
        if not text:
            continue
        words.append({"word": text, "start": start, "end": end})
    return words


# ---------------------------------------------------------------------
# Paso 4 (opcional): corregir contra la letra real
# ---------------------------------------------------------------------

def normalize(word: str) -> str:
    w = word.lower().strip()
    w = re.sub(r"[^\w\sáéíóúñü']", "", w, flags=re.UNICODE)
    return w


def load_lyrics_words(lyrics_path: Path):
    text = lyrics_path.read_text(encoding="utf-8")
    raw_words = text.split()
    return raw_words


def correct_with_lyrics(whisper_words, lyrics_words):
    """Alinea la salida de whisper (con timing) contra la letra real
    (sin timing) usando difflib. Devuelve una lista de dicts
    {"word", "start", "end"} usando el texto correcto de la letra
    pero el timing derivado de whisper.

    - 'equal'/'replace': se usa la palabra de la letra con el timing
      de whisper correspondiente (corrige errores de transcripción
      y alucinaciones manteniendo el timing real).
    - 'delete' (whisper dijo algo que no está en la letra): se
      descarta -> probable alucinación o repetición.
    - 'insert' (la letra tiene una palabra que whisper no captó): se
      interpola el timing entre la palabra anterior y la siguiente
      con timing conocido.
    """
    w_norm = [normalize(w["word"]) for w in whisper_words]
    l_norm = [normalize(w) for w in lyrics_words]

    sm = difflib.SequenceMatcher(a=w_norm, b=l_norm, autojunk=False)
    result = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            n = min(i2 - i1, j2 - j1)
            # empareja 1 a 1 lo que se pueda; si hay descuadre de
            # longitud, el resto cae en insert/delete más abajo
            for k in range(n):
                src = whisper_words[i1 + k]
                result.append({
                    "word": lyrics_words[j1 + k],
                    "start": src["start"],
                    "end": src["end"],
                })
            # sobras de la letra sin pareja directa -> se interpolarán
            for k in range(j1 + n, j2):
                result.append({"word": lyrics_words[k], "start": None, "end": None})
            # sobras de whisper sin pareja -> se descartan (alucinación)
        elif tag == "insert":
            for k in range(j1, j2):
                result.append({"word": lyrics_words[k], "start": None, "end": None})
        elif tag == "delete":
            pass  # whisper dijo palabras que no están en la letra: se ignoran

    # interpolar los None usando los vecinos con timing conocido
    n = len(result)
    for idx, item in enumerate(result):
        if item["start"] is not None:
            continue
        prev_end = None
        for k in range(idx - 1, -1, -1):
            if result[k]["end"] is not None:
                prev_end = result[k]["end"]
                break
        next_start = None
        for k in range(idx + 1, n):
            if result[k]["start"] is not None:
                next_start = result[k]["start"]
                break
        if prev_end is None:
            prev_end = 0.0
        if next_start is None:
            next_start = prev_end + 0.4
        item["start"] = prev_end
        item["end"] = min(next_start, prev_end + 0.4)

    return result


# ---------------------------------------------------------------------
# Paso 5: agrupar palabras en líneas de subtítulo y escribir el .srt
# ---------------------------------------------------------------------

def group_into_lines(words):
    lines = []
    current = []
    current_start = None

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
    """Devuelve (video_path, lyrics_path_or_None, output_path)."""
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
            # ARG2 = letra real
            return video_path, arg2.resolve(), default_output
        # ARG2 = destino de salida (carpeta o fichero)
        output = resolve_output_path(arg2, video_path)
        return video_path, None, output

    # 3 argumentos: video, letra, salida
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
    # se trata como directorio (exista ya o no)
    raw.mkdir(parents=True, exist_ok=True)
    return (raw / (video_path.stem + ".srt")).resolve()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    check_deps()
    video_path, lyrics_path, output_path = resolve_args(sys.argv[1:])

    print(f"Vídeo:   {video_path}")
    print(f"Letra:   {lyrics_path if lyrics_path else '(no proporcionada — sin corrección)'}")
    print(f"Salida:  {output_path}")

    with tempfile.TemporaryDirectory(prefix="subs_letras_") as tmp:
        workdir = Path(tmp)

        audio_path = extract_audio(video_path, workdir)
        vocals_path = separate_vocals(audio_path, workdir)
        vocals_16k = to_whisper_wav(vocals_path, workdir)
        words = transcribe_words(vocals_16k, workdir)

        if not words:
            print("Whisper no detectó ninguna palabra. Revisa el audio separado:")
            print(f"  {vocals_path}")
            sys.exit(1)

        if lyrics_path:
            print("--> Corrigiendo transcripción contra la letra real...")
            lyrics_words = load_lyrics_words(lyrics_path)
            words = correct_with_lyrics(words, lyrics_words)

        lines = group_into_lines(words)
        write_srt(lines, output_path)

    print(f"\n✅ Subtítulos generados: {output_path}")


if __name__ == "__main__":
    main()
