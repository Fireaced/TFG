# Validación del corrector de nombres — 13/10/2026 15:52

Porcentajes sobre los nombres bien leídos («bien: …») o sobre los mal leídos («mal: …») de cada columna.

## A) Errores reales de Donut (donut_lineas_20261013_1532.csv, 1024 líneas emparejadas con CER ≤ 0,5, sin tickets de entrenamiento)

### Resultado por tipo de imagen y vocabulario

| Resultado | PDF · texto del PDF | PDF · historial | PDF · catálogo | foto simulada · historial | foto simulada · catálogo |
|---|---|---|---|---|---|
| bien: sin tocar | 100.0 % (294) | 99.7 % (293) | 99.0 % (291) | 100.0 % (468) | 99.6 % (466) |
| bien: cambiado (falsa corrección) | 0.0 % (0) | 0.3 % (1) | 1.0 % (3) | 0.0 % (0) | 0.4 % (2) |
| mal: corregido bien | 64.3 % (9) | 35.7 % (5) | 28.6 % (4) | 36.7 % (91) | 23.8 % (59) |
| mal: corregido a medias (mejora) | 0.0 % (0) | 0.0 % (0) | 0.0 % (0) | 2.4 % (6) | 2.4 % (6) |
| mal: corregido mal (no mejora) | 0.0 % (0) | 7.1 % (1) | 7.1 % (1) | 2.0 % (5) | 3.6 % (9) |
| mal: sin corregir | 35.7 % (5) | 57.1 % (8) | 64.3 % (9) | 58.9 % (146) | 70.2 % (174) |
| Nombres | 308 | 308 | 308 | 716 | 716 |

En la app, al subir un PDF se usa la columna «texto del PDF» (nombres exactos del propio ticket); las otras dos columnas de PDF muestran qué pasaría sin ese texto.

### Fotos simuladas, según lo grave que sea el error de lectura

| Resultado | foto · historial · error leve (CER ≤ 0,2) | foto · historial · error grave (CER > 0,2) |
|---|---|---|
| bien: sin tocar | 100.0 % (4) | — |
| mal: corregido bien | 44.6 % (90) | 2.2 % (1) |
| mal: corregido a medias (mejora) | 2.0 % (4) | 4.3 % (2) |
| mal: corregido mal (no mejora) | 2.0 % (4) | 2.2 % (1) |
| mal: sin corregir | 51.5 % (104) | 91.3 % (42) |
| Nombres | 206 | 46 |

**mal: corregido mal (no mejora)** (6): TARTALETES FRESA. ← «TARTALLETES FRESA.» → TARTALETAS FRESA.; Q RALLADO FUNDIR ← «QALLADO FUNDIR» → RALLADO FUNDIR; CUQUIS ← «CHOUS» → COUS; SALCHICHON 4PACK ← «SALCHICHON SPACK» → SALCHICHON PACK; BERBERECHOS XS ← «BERBERECHONXS» → BERBERECHOS

**bien: cambiado (falsa corrección)** (1): PARMIGIANA ← «PARMIGIANA» → PARMIGIANO

## B) Errores simulados de tipo OCR (91 tickets, dejando fuera cada ticket)

### Nombres con un error simulado

| Resultado | historial · con error | catálogo · con error | PDF · con error |
|---|---|---|---|
| mal: corregido bien | 80.1 % (474) | 51.0 % (302) | 92.1 % (545) |
| mal: corregido a medias (mejora) | 1.2 % (7) | 1.0 % (6) | 1.0 % (6) |
| mal: sugerencia correcta | 0.0 % (0) | 0.5 % (3) | 0.0 % (0) |
| mal: sugerencia incorrecta | 0.0 % (0) | 0.3 % (2) | 0.0 % (0) |
| mal: corregido mal (no mejora) | 0.8 % (5) | 2.7 % (16) | 0.0 % (0) |
| mal: sin corregir | 17.9 % (106) | 44.4 % (263) | 6.9 % (41) |
| Nombres | 592 | 592 | 592 |

### Nombres correctos (¿los estropea?)

| Resultado | historial · sin error | catálogo · sin error | PDF · sin error |
|---|---|---|---|
| bien: sin tocar | 99.8 % (591) | 98.6 % (584) | 100.0 % (592) |
| bien: solo sugerencia | 0.0 % (0) | 0.7 % (4) | 0.0 % (0) |
| bien: cambiado (falsa corrección) | 0.2 % (1) | 0.7 % (4) | 0.0 % (0) |
| Nombres | 592 | 592 | 592 |

### Con historial: producto ya comprado antes vs producto nuevo

| Resultado | historial · producto conocido · con error | historial · producto nuevo · con error |
|---|---|---|
| mal: corregido bien | 91.8 % (391) | 50.0 % (83) |
| mal: corregido a medias (mejora) | 1.4 % (6) | 0.6 % (1) |
| mal: corregido mal (no mejora) | 0.0 % (0) | 3.0 % (5) |
| mal: sin corregir | 6.8 % (29) | 46.4 % (77) |
| Nombres | 426 | 166 |

**simulado: corregido mal** (5): CHAMPÚ FAMILY ← «CMAMPÚ FAMILY» → CHAMPU FAMIL; BERBERECHO S 65/100 ← «BERBFRECHO S 65/100» → BERBERECHOS 65/100; PAT. CLASSICAS ← «PAT. CLASSSICAS» → PAT. CLASICAS; PATATAS CAMPESTRES ← «PTATAS CAMPESTRES» → PATATAS CAMPESTRE; CHORIZO PICANT 4 PAC ← «CHORIZO PICAÑT 4 PAC» → CHORIZO PICANTE 4 PAC

**simulado: falsa corrección** (1): PAT. CLASSICAS ← «PAT. CLASSICAS» → PAT. CLASICAS

