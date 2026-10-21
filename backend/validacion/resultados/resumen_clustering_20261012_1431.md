# Clasificación de productos mediante clustering — 12/10/2026 15:36

Catálogo: 3931 productos (26 categorías, 150 subcategorías) · Tickets: 280 productos con categoría

## A. catálogo (CV) — acierto de categoría (subcategoría)

| Método | tfidf_char | tfidf_palabras | tfidf_mixto | emb:bge-m3 | emb:nomic-embed-text | emb:bge-m3+char | emb:nomic-embed-text+char |
|---|---|---|---|---|---|---|---|
| kmeans (k=26) | 48.6 % (21.5 %) | 46.1 % (21.6 %) | 50.7 % (22.2 %) | 55.2 % (23.6 %) | 44.4 % (20.9 %) | 52.2 % (22.4 %) | 47.5 % (21.6 %) |
| kmeans (k=148) | 66.9 % (47.6 %) | 68.9 % (49.5 %) | 68.6 % (49.8 %) | 71.7 % (49.3 %) | 65.0 % (45.5 %) | 71.7 % (51.5 %) | 68.6 % (48.8 %) |
| kmeans (k=444) | 75.6 % (61.4 %) | 76.4 % (61.4 %) | 77.6 % (62.6 %) | 81.0 % (63.7 %) | 75.0 % (58.2 %) | 79.8 % (65.6 %) | 77.8 % (63.2 %) |
| jerarquico (k=148) | 67.5 % (47.4 %) | 68.2 % (51.0 %) | 68.7 % (50.4 %) | 73.4 % (50.7 %) | 67.3 % (46.2 %) | 73.0 % (51.9 %) | 70.0 % (49.7 %) |
| jerarquico (k=444) | 75.7 % (61.6 %) | 76.2 % (62.6 %) | 77.5 % (63.2 %) | 81.0 % (65.8 %) | 75.0 % (60.0 %) | 80.6 % (66.2 %) | 78.0 % (64.6 %) |
| hdbscan (min=3) | 75.6 % (60.7 %) | 75.6 % (61.9 %) | 76.4 % (62.2 %) | 79.8 % (61.8 %) | 73.2 % (55.8 %) | 80.1 % (64.7 %) | 77.2 % (62.5 %) |
| 1-NN | 84.5 % (74.6 %) | 82.5 % (71.5 %) | 85.2 % (75.3 %) | 88.9 % (78.4 %) | 85.5 % (74.4 %) | 88.5 % (78.9 %) | 86.9 % (77.0 %) |
| 5-NN | 82.7 % (72.1 %) | 82.9 % (71.7 %) | 84.3 % (73.4 %) | 88.5 % (78.1 %) | 84.6 % (74.3 %) | 86.5 % (76.9 %) | 84.4 % (74.7 %) |

## B. catálogo → tickets — acierto de categoría (subcategoría)

| Método | tfidf_char | tfidf_palabras | tfidf_mixto | emb:bge-m3 | emb:nomic-embed-text | emb:bge-m3+char | emb:nomic-embed-text+char |
|---|---|---|---|---|---|---|---|
| kmeans (k=26) | 32.9 % (14.6 %) | 32.9 % (9.6 %) | 28.9 % (10.7 %) | 17.1 % (4.6 %) | 17.5 % (8.6 %) | 21.8 % (9.6 %) | 20.4 % (7.5 %) |
| kmeans (k=148) | 44.6 % (35.0 %) | 52.5 % (36.8 %) | 51.8 % (40.4 %) | 35.0 % (21.4 %) | 33.6 % (23.2 %) | 51.1 % (38.2 %) | 45.4 % (33.6 %) |
| kmeans (k=444) | 63.9 % (55.4 %) | 56.4 % (46.4 %) | 59.3 % (53.2 %) | 53.9 % (43.6 %) | 53.9 % (43.2 %) | 61.8 % (52.9 %) | 58.2 % (51.1 %) |
| jerarquico (k=148) | 48.9 % (38.2 %) | 51.8 % (35.7 %) | 54.3 % (40.7 %) | 37.1 % (25.0 %) | 31.8 % (25.0 %) | 48.2 % (37.1 %) | 46.4 % (38.9 %) |
| jerarquico (k=444) | 58.6 % (51.8 %) | 61.4 % (51.4 %) | 63.2 % (53.6 %) | 52.5 % (41.4 %) | 53.6 % (45.4 %) | 60.0 % (53.2 %) | 59.6 % (53.2 %) |
| hdbscan (min=3) | 61.4 % (54.6 %) | 58.9 % (50.7 %) | 61.8 % (54.3 %) | 59.6 % (51.1 %) | 53.2 % (43.9 %) | 64.3 % (57.5 %) | 61.1 % (54.6 %) |
| 1-NN | 77.5 % (73.2 %) | 68.6 % (60.4 %) | 78.9 % (74.3 %) | 82.5 % (78.6 %) | 80.0 % (76.1 %) | 87.5 % (83.6 %) | 84.6 % (80.4 %) |
| 5-NN | 72.1 % (66.4 %) | 65.7 % (59.6 %) | 76.4 % (72.5 %) | 75.7 % (73.2 %) | 73.2 % (66.1 %) | 77.5 % (73.6 %) | 76.1 % (70.4 %) |
| *Buscador por palabras del catálogo (actual)* | 92.1 % (91.1 %), responde en el 96.8 % |  |  |  |  |  |  |

## C. tickets (CV, + catálogo) — acierto de categoría (subcategoría)

| Método | tfidf_char | tfidf_palabras | tfidf_mixto | emb:bge-m3 | emb:nomic-embed-text | emb:bge-m3+char | emb:nomic-embed-text+char |
|---|---|---|---|---|---|---|---|
| kmeans (k=26) | 27.1 % (13.6 %) | 35.0 % (11.8 %) | 31.4 % (12.5 %) | 21.1 % (6.8 %) | 17.5 % (8.2 %) | 19.6 % (6.8 %) | 20.4 % (8.2 %) |
| kmeans (k=148) | 47.9 % (35.7 %) | 52.1 % (33.9 %) | 49.3 % (36.8 %) | 36.8 % (21.8 %) | 33.6 % (24.6 %) | 46.4 % (35.7 %) | 48.2 % (36.1 %) |
| kmeans (k=444) | 55.0 % (49.3 %) | 58.6 % (50.7 %) | 61.1 % (50.7 %) | 50.7 % (40.0 %) | 52.5 % (42.5 %) | 65.0 % (56.4 %) | 61.4 % (53.9 %) |
| jerarquico (k=148) | 47.1 % (37.1 %) | 49.6 % (37.9 %) | 52.5 % (37.5 %) | 36.8 % (26.4 %) | 34.6 % (27.1 %) | 46.8 % (35.0 %) | 44.6 % (33.9 %) |
| jerarquico (k=444) | 59.6 % (53.2 %) | 62.1 % (52.9 %) | 62.1 % (53.9 %) | 56.1 % (41.8 %) | 53.6 % (44.3 %) | 60.0 % (52.9 %) | 56.1 % (50.0 %) |
| hdbscan (min=3) | 63.2 % (56.1 %) | 57.5 % (48.9 %) | 60.4 % (52.5 %) | 61.8 % (52.1 %) | 53.6 % (44.6 %) | 65.0 % (58.2 %) | 62.1 % (55.0 %) |
| 1-NN | 77.9 % (71.4 %) | 71.8 % (62.9 %) | 81.8 % (74.3 %) | 84.3 % (75.7 %) | 83.6 % (77.9 %) | 86.1 % (78.9 %) | 85.4 % (79.6 %) |
| 5-NN | 72.9 % (66.8 %) | 66.4 % (61.1 %) | 76.1 % (71.8 %) | 80.4 % (74.3 %) | 76.1 % (68.9 %) | 79.3 % (73.9 %) | 77.9 % (71.4 %) |
| *Buscador por palabras del catálogo (actual)* | 92.1 % (91.1 %), responde en el 96.8 % |  |  |  |  |  |  |

## Calidad del agrupamiento sin etiquetas (k = nº de categorías)

| Representación | Datos | Algoritmo | Silueta | ARI | NMI |
|---|---|---|---|---|---|
| tfidf_char | catalogo | kmeans | 0.141 | 0.223 | 0.527 |
| tfidf_char | catalogo | jerarquico | 0.094 | 0.199 | 0.496 |
| tfidf_char | tickets | kmeans | 0.124 | 0.090 | 0.410 |
| tfidf_char | tickets | jerarquico | 0.141 | 0.016 | 0.357 |
| tfidf_palabras | catalogo | kmeans | 0.145 | 0.198 | 0.483 |
| tfidf_palabras | catalogo | jerarquico | 0.080 | 0.114 | 0.467 |
| tfidf_palabras | tickets | kmeans | 0.164 | 0.061 | 0.389 |
| tfidf_palabras | tickets | jerarquico | 0.200 | 0.013 | 0.368 |
| tfidf_mixto | catalogo | kmeans | 0.117 | 0.261 | 0.523 |
| tfidf_mixto | catalogo | jerarquico | 0.079 | 0.177 | 0.494 |
| tfidf_mixto | tickets | kmeans | 0.132 | 0.055 | 0.372 |
| tfidf_mixto | tickets | jerarquico | 0.149 | 0.020 | 0.377 |
| emb:bge-m3 | catalogo | kmeans | 0.082 | 0.319 | 0.565 |
| emb:bge-m3 | catalogo | jerarquico | 0.060 | 0.320 | 0.588 |
| emb:bge-m3 | tickets | kmeans | 0.048 | 0.086 | 0.405 |
| emb:bge-m3 | tickets | jerarquico | 0.061 | 0.110 | 0.443 |
| emb:nomic-embed-text | catalogo | kmeans | 0.088 | 0.263 | 0.466 |
| emb:nomic-embed-text | catalogo | jerarquico | 0.069 | 0.236 | 0.497 |
| emb:nomic-embed-text | tickets | kmeans | 0.042 | 0.072 | 0.389 |
| emb:nomic-embed-text | tickets | jerarquico | 0.061 | 0.060 | 0.372 |
| emb:bge-m3+char | catalogo | kmeans | 0.123 | 0.274 | 0.550 |
| emb:bge-m3+char | catalogo | jerarquico | 0.074 | 0.276 | 0.577 |
| emb:bge-m3+char | tickets | kmeans | 0.071 | 0.093 | 0.398 |
| emb:bge-m3+char | tickets | jerarquico | 0.092 | 0.059 | 0.421 |
| emb:nomic-embed-text+char | catalogo | kmeans | 0.135 | 0.281 | 0.537 |
| emb:nomic-embed-text+char | catalogo | jerarquico | 0.081 | 0.206 | 0.536 |
| emb:nomic-embed-text+char | tickets | kmeans | 0.083 | 0.071 | 0.391 |
| emb:nomic-embed-text+char | tickets | jerarquico | 0.111 | 0.041 | 0.390 |

## Mejor configuración por escenario

- **A. catálogo (CV)**: clustering → emb:bge-m3 + kmeans (k=444): 81.0 % (subcat. 63.7 %); referencia k-NN → emb:bge-m3 + 1-NN: 88.9 %
- **B. catálogo → tickets**: clustering → emb:bge-m3+char + hdbscan (min=3): 64.3 % (subcat. 57.5 %); referencia k-NN → emb:bge-m3+char + 1-NN: 87.5 %
- **C. tickets (CV, + catálogo)**: clustering → emb:bge-m3+char + kmeans (k=444): 65.0 % (subcat. 56.4 %); referencia k-NN → emb:bge-m3+char + 1-NN: 86.1 %

Notas: «acierto» = la categoría asignada está entre las del producto (un producto puede tener varias). Las etiquetas de los tickets son las de etiquetas_tickets.csv (revisadas a mano), independientes del buscador.
