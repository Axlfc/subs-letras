# subs-letras — subtítulos de vídeos musicales con Demucs + whisper.cpp

Genera subtítulos `.srt` de un vídeo musical (p. ej. un lyric video del
LP de EL AFACES): separa la voz de la música con **Demucs**, transcribe
la voz aislada con **whisper.cpp** (timings palabra a palabra), y si le
pasas la letra real, corrige las alucinaciones de Whisper manteniendo
el timing correcto.

## Por qué separar la voz primero

Whisper transcribe mucho peor (y alucina más) cuando hay instrumental
de fondo mezclado con la voz — es exactamente el mismo problema de
alucinación por audio "sucio" que vimos con el dictado por voz, pero
aquí en vez de silencio es música. Separar con Demucs antes de
transcribir mejora bastante la precisión y reduce las alucinaciones
antes incluso de aplicar la corrección por letra.

## Instalación

```bash
pip install --break-system-packages -r requirements.txt
mkdir -p ~/.config/scripts
cp subs_letras.py ~/.config/scripts/
cp subsletra.fish ~/.config/fish/functions/   # si usas fish functions así
```

Necesitas whisper.cpp ya compilado (el mismo que usas para el dictado
por voz) y `ffmpeg`. El script reutiliza por defecto las mismas rutas
del proyecto de dictado (`~/whisper.cpp/build/bin/whisper-cli` y el
modelo `large-v3-turbo`); sobreescríbelas con las variables de entorno
`WHISPER_BIN` / `WHISPER_MODEL` / `DICTADO_LANG` si usas otras.

La primera vez que ejecutes Demucs, descargará el modelo
`htdemucs` (~80 MB) automáticamente.

## Uso

```bash
# 1 argumento: solo el vídeo -> genera video.srt al lado del vídeo
subsletra video.mp4

# 2 argumentos, letra real (.txt existente) -> corrige alucinaciones,
# salida por defecto (video.srt al lado del vídeo)
subsletra video.mp4 letra.txt

# 2 argumentos, ruta de salida (no es un .txt existente) -> sin
# corrección de letra, pero exporta donde le digas
subsletra video.mp4 /home/axel/subtitulos/
subsletra video.mp4 /home/axel/subtitulos/mi_cancion.srt

# 3 argumentos: letra + salida explícitos
subsletra video.mp4 letra.txt /home/axel/subtitulos/mi_cancion.srt
```

La detección de si el segundo argumento es "letra" o "salida" se basa
en si es un `.txt` que existe. Si tu ruta de salida por casualidad
también termina en `.txt` en vez de `.srt`, cámbiale la extensión o
pasa los 3 argumentos explícitos para evitar ambigüedad.

## Cómo funciona la corrección con letra real

1. Whisper transcribe la voz aislada palabra por palabra, con su
   timing.
2. Se compara esa transcripción contra las palabras de tu `.txt`
   usando un diff de secuencias (`difflib`).
3. Donde coinciden (o casi): se usa la palabra **correcta** de tu
   letra, pero con el **timing real** que detectó Whisper — así
   arreglas errores de transcripción sin romper la sincronía.
4. Palabras que Whisper "dijo" pero no están en tu letra → se
   descartan (alucinación).
5. Palabras de tu letra que Whisper no captó → se interpola su
   timing entre las palabras vecinas que sí tienen timing conocido.

Esto es una heurística, no un alineamiento forzado "de verdad" (como
haría Montreal Forced Aligner). Funciona bien cuando el orden de las
palabras coincide razonablemente con la letra real — que es el caso
normal salvo que Whisper se pierda por completo en algún tramo. Si un
tramo entero sale muy descuadrado en el `.srt` final, revísalo a mano;
la interpolación en esos casos es solo una aproximación.

## Ajustes

En `subs_letras.py`:
- `MAX_LINE_CHARS` — caracteres máximos por línea de subtítulo (42 por
  defecto, estándar razonable para YouTube)
- `MAX_LINE_SECONDS` — duración máxima de un bloque
- `GAP_NEW_LINE` — silencio mínimo entre palabras para cortar línea
- `DEMUCS_MODEL` — modelo de Demucs (`htdemucs` por defecto; hay
  variantes como `htdemucs_ft` más lenta pero algo más precisa)

## Notas sobre flags de whisper.cpp

El script usa `-ml 1 -sow -osrt -of <prefijo>` para forzar timings
palabra a palabra. Estos nombres de flag pueden variar ligeramente
entre versiones de whisper.cpp — si el script falla al no encontrar el
`.srt` generado, ejecuta `whisper-cli --help` y ajusta los flags en la
función `transcribe_words()`.
