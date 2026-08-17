# Código reproducible: FFT Strang para el sistema GDSS

Esta carpeta reúne el código y los materiales de reproducibilidad del artículo
**“A reproducible FFT-based Strang-splitting baseline for the generalized
Davey--Stewartson system”**, preparado para la revista *International Journal of
Modern Physics C*.

Repositorio: <https://github.com/Camolina-uach/FFT---Strang---GDSS>

Los cinco archivos Python originales se conservaron sin cambios. La revisión de
versiones y sus huellas SHA-256 se documentan en
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Instalación y prueba rápida

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce.py smoke
```

La prueba `smoke` utiliza una malla pequeña y un intervalo temporal corto. Sirve
para comprobar que el entorno y el flujo de experimentos funcionan; no reproduce
los valores finales del artículo.

## Reproducción completa

```bash
python reproduce.py paper
python scripts/compare_reference_tables.py
```

Los resultados se guardan en `outputs/`. La exportación completa para ParaView
es grande y por eso se activa sólo cuando se solicita:

```bash
python reproduce.py paper --with-paraview
```

El comando histórico exacto usado para producir resultados, figuras y archivos
de ParaView también se conserva:

```bash
python run.py
```

Ambos comandos completos pueden reemplazar archivos con el mismo nombre dentro
de `outputs/`; conviene respaldar una corrida que se desee conservar.

Las ocho tablas CSV incorporadas al manuscrito están en
`reference_results/paper_tables/`. El comparador omite los tiempos de cómputo,
porque cambian con el equipo, y evalúa los valores científicos con tolerancias
numéricas documentadas.

Antes de publicar la carpeta en GitHub queda una verificación editorial:

1. Añadir a `CITATION.cff` el DOI del artículo y el DOI del archivo de software
   (por ejemplo, el asignado por Zenodo).

La publicación del código con la licencia permisiva BSD-3-Clause incluida en
`LICENSE` ya fue autorizada por los titulares de los derechos.
