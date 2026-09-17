# subs-letras

**Generación local de subtítulos sincronizados para vídeos musicales usando Demucs, whisper.cpp y forced alignment.**

`subs-letras` toma un vídeo musical —por ejemplo un `.mp4` de un LP completo—, separa la voz de la música y genera un fichero `.srt` listo para importar en YouTube.

Está pensado especialmente para material musical, spoken word, lyric videos y álbumes completos donde una transcripción directa con Whisper puede degradarse debido a la presencia de música, efectos, ruido, reverberación o voces procesadas.

Todo el procesamiento puede ejecutarse localmente.

---

## Qué hace

El pipeline tiene dos modos distintos dependiendo de si dispones o no de la letra original.

### Sin letra

```text
MP4
 │
 ▼
FFmpeg
 │
 ▼
audio.wav
 │
 ▼
Demucs
 │
 ▼
vocals.wav
 │
 ▼
whisper.cpp
large-v3 + DTW + Silero VAD
 │
 ▼
timings + transcripción
 │
 ▼
SRT
```

Whisper transcribe la voz aislada y genera timestamps a nivel de palabra.

Silero VAD ayuda a detectar las regiones donde realmente existe voz y reduce el riesgo de que Whisper genere texto durante silencios, instrumentales o zonas con muy poca información vocal.

---

### Con letra original

```text
MP4
 │
 ▼
FFmpeg
 │
 ▼
audio.wav
 │
 ▼
Demucs
 │
 ▼
vocals.wav
 │
 ├──────────── lyrics.txt
 │                 │
 ▼                 ▼
       CTC Forced Alignment
        MMS multilingual
                │
                ▼
      letra exacta + timings
                │
                ▼
               SRT
```

En este modo **Whisper no transcribe la letra**.

La letra proporcionada se considera el *ground truth* y un modelo de forced alignment calcula cuándo ocurre cada palabra en el audio.

Esto tiene una ventaja importante:

> si ya sabemos exactamente qué se está diciendo, no necesitamos pedirle a una IA que vuelva a adivinarlo.

Por tanto, errores o alucinaciones de una transcripción automática no pueden modificar el texto final.

---

## Stack

El proyecto utiliza:

- **FFmpeg** — extracción y conversión del audio.
- **Demucs** — separación `vocals / instrumental`.
- **whisper.cpp** — transcripción cuando no existe letra.
- **Whisper large-v3** — modelo ASR utilizado por defecto.
- **Silero VAD** — detección de actividad vocal para el modo Whisper.
- **DTW de whisper.cpp** — timestamps de mayor resolución.
- **ctc-forced-aligner** — alineamiento cuando existe una letra conocida.
- **MMS Forced Aligner** — modelo multilingual utilizado para forced alignment.
- **PyTorch / CUDA** — aceleración GPU cuando está disponible.

---

# Instalación

El proyecto está pensado principalmente para Linux y ha sido desarrollado sobre **CachyOS / Arch Linux**.

## 1. Clonar el repositorio

```bash
git clone https://github.com/Axlfc/subs-letras.git
cd subs-letras
```

---

## 2. Dependencias del sistema

En CachyOS / Arch:

```bash
sudo pacman -S --needed \
    git \
    ffmpeg \
    cmake \
    python \
    python-pip
```

Para aceleración NVIDIA/CUDA:

```bash
sudo pacman -S --needed cuda
```

---

## 3. Instalar las dependencias Python

```bash
pip install --break-system-packages -r requirements.txt
```

Actualmente se utilizan, entre otras:

```text
demucs
torch
torchaudio
ctc-forced-aligner
```

La primera vez que se utilice Demucs descargará automáticamente su modelo.

El modelo por defecto es:

```text
htdemucs
```

---

# Instalar whisper.cpp

`whisper.cpp` solo es obligatorio para el modo **sin lyrics**.

Si siempre vas a proporcionar la letra correcta, el forced alignment puede funcionar sin realizar una transcripción Whisper.

Para disponer de ambos modos se recomienda instalarlo.

## 4. Clonar whisper.cpp

```bash
cd ~
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

---

## 5. Compilar whisper.cpp con CUDA

Para sistemas con una GPU NVIDIA:

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build -j --config Release
```

El ejecutable esperado por defecto es:

```text
~/whisper.cpp/build/bin/whisper-cli
```

Compruébalo con:

```bash
~/whisper.cpp/build/bin/whisper-cli --help
```

---

# Descargar Whisper large-v3

El modo sin letra utiliza por defecto **Whisper large-v3**.

Desde `~/whisper.cpp`:

```bash
cd ~/whisper.cpp
bash ./models/download-ggml-model.sh large-v3
```

El modelo quedará normalmente en:

```text
~/whisper.cpp/models/ggml-large-v3.bin
```

Puedes comprobarlo con:

```bash
ls -lh ~/whisper.cpp/models/ggml-large-v3.bin
```

---

# Descargar Silero VAD

Para reducir transcripciones falsas durante silencios o secciones instrumentales se utiliza **Silero VAD**.

Desde el repositorio de `whisper.cpp`:

```bash
cd ~/whisper.cpp
bash ./models/download-vad-model.sh silero-v5.1.2
```

El fichero esperado es:

```text
~/whisper.cpp/models/ggml-silero-v5.1.2.bin
```

Compruébalo con:

```bash
ls -lh ~/whisper.cpp/models/ggml-silero-v5.1.2.bin
```

> Si el modelo VAD no está disponible, el script puede continuar con Whisper, pero habrá mayor riesgo de transcripciones espurias en silencios o zonas instrumentales.

---

# Instalar el comando `subsletra`

Copia el script:

```bash
mkdir -p ~/.config/scripts

cp subs_letras.py ~/.config/scripts/subs_letras.py
chmod +x ~/.config/scripts/subs_letras.py
```

Instala la función Fish:

```bash
mkdir -p ~/.config/fish/functions
cp subsletra.fish ~/.config/fish/functions/subsletra.fish
```

Recarga Fish:

```fish
source ~/.config/fish/config.fish
```

o simplemente abre una terminal nueva.

Comprueba el comando:

```fish
type subsletra
```

---

# Uso

## 1 argumento — vídeo

```bash
subsletra video.mp4
```

Genera:

```text
video.mp4
video.srt
```

en el mismo directorio.

Como no existe letra de referencia, el pipeline utilizado es:

```text
Demucs → Whisper large-v3 → DTW/VAD → SRT
```

---

## 2 argumentos — vídeo + letra

Si el segundo argumento es un `.txt` existente:

```bash
subsletra video.mp4 lyrics.txt
```

se interpreta como letra oficial.

Genera:

```text
video.srt
```

junto al vídeo.

El pipeline utilizado es:

```text
Demucs → MMS Forced Alignment → SRT
```

La letra del `.txt` será el texto utilizado en los subtítulos.

---

## 2 argumentos — vídeo + directorio de salida

```bash
subsletra video.mp4 /home/usuario/subtitulos/
```

Generará:

```text
/home/usuario/subtitulos/video.srt
```

---

## 2 argumentos — vídeo + fichero de salida

```bash
subsletra video.mp4 /home/usuario/subtitulos/album.srt
```

Generará exactamente:

```text
/home/usuario/subtitulos/album.srt
```

---

## 3 argumentos — vídeo + letra + salida

```bash
subsletra video.mp4 lyrics.txt /home/usuario/subtitulos/album.srt
```

Este es el modo más explícito y recomendable para álbumes completos:

```text
ARG1 = vídeo
ARG2 = lyrics.txt
ARG3 = output
```

Ejemplo:

```bash
subsletra \
    "EL AFACES - ENTROPÍA BUROCRÁTICA.mp4" \
    "/home/axel/Música/96/lyrics.txt" \
    "./EL AFACES - ENTROPÍA BUROCRÁTICA.srt"
```

---

# Resolución de argumentos

La interfaz está diseñada para mantener el comando simple:

| Argumentos | Interpretación |
|---|---|
| `video.mp4` | vídeo → SRT automático |
| `video.mp4 lyrics.txt` | vídeo + letra |
| `video.mp4 output/` | vídeo + directorio de salida |
| `video.mp4 output.srt` | vídeo + fichero de salida |
| `video.mp4 lyrics.txt output.srt` | vídeo + letra + salida |

Con dos argumentos, un fichero `.txt` existente se interpreta como letra.

Para eliminar cualquier ambigüedad puedes utilizar siempre la forma de tres argumentos.

---

# Formato de salida

La salida es un fichero estándar **SubRip `.srt`**:

```srt
1
00:00:12,420 --> 00:00:15,180
La primera línea de la letra

2
00:00:15,420 --> 00:00:18,930
continúa exactamente aquí
```

Aunque `.srt` es técnicamente texto plano, es preferible mantener la extensión `.srt` porque puede importarse directamente como fichero de subtítulos en plataformas como YouTube y en editores de vídeo compatibles.

---

# Por qué Demucs

Los modelos ASR como Whisper están diseñados principalmente para reconocer voz.

En un master musical pueden coexistir:

- voz,
- batería,
- bajos,
- sintetizadores,
- guitarras,
- reverberación,
- distorsión,
- ruido,
- samples,
- efectos creativos.

Demucs intenta separar primero:

```text
master
├── vocals
└── no_vocals / instrumental
```

El sistema trabaja después principalmente con el stem vocal.

Eso proporciona al sistema de alineamiento o transcripción una señal considerablemente más limpia que el master musical original.

---

# Whisper, VAD y alucinaciones

Whisper puede producir texto plausible incluso cuando la señal no contiene una voz inteligible.

En música esto puede suceder especialmente durante:

- intros,
- outros,
- silencios,
- drones,
- secciones instrumentales,
- ruido ambiental,
- reverberaciones largas,
- samples muy degradados.

Por ese motivo el modo de transcripción combina:

```text
Demucs
   ↓
vocals
   ↓
Silero VAD
   ↓
Whisper large-v3
   ↓
DTW / timestamps
```

VAD reduce la cantidad de audio sin actividad vocal que llega al proceso de reconocimiento.

No elimina matemáticamente todas las posibles alucinaciones, pero reduce uno de sus principales escenarios de aparición.

---

# Forced alignment con letra real

Cuando proporcionas `lyrics.txt`, el problema cambia completamente.

No necesitamos resolver:

```text
¿qué está diciendo?
```

porque ya conocemos la respuesta.

Solo necesitamos resolver:

```text
¿cuándo está diciendo cada palabra?
```

El script utiliza:

```text
MahmoudAshraf/mms-300m-1130-forced-aligner
```

para alinear el texto proporcionado con el audio de voz separado por Demucs.

El flujo queda:

```text
lyrics.txt ───────────────┐
                         │
MP4 → FFmpeg → Demucs ───┼→ forced alignment → timestamps
                         │
                         └→ texto conocido
```

Esto evita que una palabra incorrectamente reconocida por Whisper termine sustituyendo a la letra original.

---

# Idioma del forced alignment

Por defecto:

```text
ALIGN_LANG_ISO3=spa
```

corresponde a español.

Puedes cambiarlo mediante variable de entorno.

Ejemplo para inglés:

```bash
export ALIGN_LANG_ISO3=eng
```

Para catalán:

```bash
export ALIGN_LANG_ISO3=cat
```

---

# Configuración mediante variables de entorno

El comportamiento puede modificarse sin editar el código:

```bash
export WHISPER_BIN="$HOME/whisper.cpp/build/bin/whisper-cli"
export WHISPER_MODEL="$HOME/whisper.cpp/models/ggml-large-v3.bin"
export DICTADO_LANG="es"

export WHISPER_DTW="large.v3"
export WHISPER_VAD_MODEL="$HOME/whisper.cpp/models/ggml-silero-v5.1.2.bin"

export DEMUCS_MODEL="htdemucs"

export ALIGN_LANG_ISO3="spa"
export ALIGN_MODEL="MahmoudAshraf/mms-300m-1130-forced-aligner"
```

Los valores anteriores corresponden a la configuración por defecto del proyecto.

---

# Ajustes de subtítulos

En `subs_letras.py` existen varios parámetros importantes:

```python
MAX_LINE_CHARS = 42
MAX_LINE_SECONDS = 6.0
GAP_NEW_LINE = 0.6
```

### `MAX_LINE_CHARS`

Número máximo aproximado de caracteres de un bloque antes de crear otro subtítulo.

### `MAX_LINE_SECONDS`

Duración máxima permitida para un bloque.

### `GAP_NEW_LINE`

Si existe un silencio superior a este valor entre dos palabras, se inicia un nuevo bloque.

---

# Modelos

## Demucs

Por defecto:

```text
htdemucs
```

Puede sustituirse mediante:

```bash
export DEMUCS_MODEL=htdemucs_ft
```

dependiendo del equilibrio deseado entre velocidad y separación.

## Whisper

Por defecto:

```text
large-v3
```

Ruta:

```text
~/whisper.cpp/models/ggml-large-v3.bin
```

## VAD

Por defecto:

```text
silero-v5.1.2
```

## Forced alignment

Por defecto:

```text
MahmoudAshraf/mms-300m-1130-forced-aligner
```

---

# GPU

El proyecto puede aprovechar CUDA en las partes compatibles del pipeline.

Para verificar PyTorch:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Para verificar `whisper.cpp`:

```bash
~/whisper.cpp/build/bin/whisper-cli --help
```

---

# Ejemplo completo

```bash
cd ~/Vídeos

subsletra \
    "EL AFACES - ENTROPÍA BUROCRÁTICA.mp4" \
    "/home/axel/Música/96/lyrics.txt" \
    "./entropia-burocratica.srt"
```

Pipeline:

```text
EL AFACES - ENTROPÍA BUROCRÁTICA.mp4
                │
                ▼
             FFmpeg
                │
                ▼
             Demucs
                │
                ▼
           vocals.wav
                │
       lyrics.txt ─────┐
                │      │
                ▼      ▼
          Forced Alignment
                │
                ▼
    entropia-burocratica.srt
```

El resultado puede importarse posteriormente como pista de subtítulos.

---

# Troubleshooting

## `whisper-cli` no existe

Comprueba:

```bash
ls ~/whisper.cpp/build/bin/whisper-cli
```

Si no existe, recompila:

```bash
cd ~/whisper.cpp

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build -j --config Release
```

---

## No existe `ggml-large-v3.bin`

```bash
cd ~/whisper.cpp
bash ./models/download-ggml-model.sh large-v3
```

---

## No existe el modelo Silero VAD

```bash
cd ~/whisper.cpp
bash ./models/download-vad-model.sh silero-v5.1.2
```

---

## Demucs tarda la primera vez

Es normal.

La primera ejecución debe descargar los modelos necesarios.

Las ejecuciones posteriores reutilizan el modelo almacenado localmente.

---

## Warning de Hugging Face sobre autenticación

Los modelos públicos pueden descargarse sin una cuenta de Hugging Face.

Una sesión autenticada puede proporcionar límites de descarga superiores, pero no es necesaria para el funcionamiento básico.

---

## Se generan subtítulos incorrectos sin lyrics

Prueba el modo con letra conocida:

```bash
subsletra video.mp4 lyrics.txt
```

De esta forma el texto del subtítulo deja de depender del reconocimiento de Whisper y se utiliza directamente la letra proporcionada.

---

# Modos resumidos

```text
SIN LETRA
═════════

MP4
 ↓
FFmpeg
 ↓
Demucs
 ↓
vocals
 ↓
Silero VAD
 ↓
Whisper large-v3 + DTW
 ↓
SRT


CON LETRA
═════════

MP4 ─→ FFmpeg ─→ Demucs ─→ vocals ─────┐
                                        │
lyrics.txt ─────────────────────────────┤
                                        ▼
                               MMS Forced Alignment
                                        │
                                        ▼
                                       SRT
```

---

## Objetivo del proyecto

`subs-letras` nace para automatizar uno de los pasos más tediosos de publicar vídeos musicales: convertir un master audiovisual y, opcionalmente, su letra original en subtítulos sincronizados utilizables sin tener que marcar manualmente cientos o miles de entradas de tiempo.

Especialmente pensado para **EL AFACES**, pero diseñado como una herramienta genérica para cualquier vídeo musical.

---

## Licencia

MIT
