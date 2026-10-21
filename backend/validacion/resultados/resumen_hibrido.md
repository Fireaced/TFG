# Categorización híbrida (buscador + respaldo) — productos de los tickets

280 productos, validación cruzada de 5 particiones, embeddings bge-m3 + TF-IDF de caracteres.

| Configuración | Acierto categoría | Productos que van al respaldo |
|---|---|---|
| Buscador del catálogo solo | 92.1 % | — (no responde en 9) |
| 1-NN embeddings solo | 86.1 % | — |
| Clustering K-Means (k=444) solo | 65.0 % | — |
| Buscador (confianza ≥ 0.55) + 1-NN | 94.6 % | 9 |
| Buscador (confianza ≥ 0.55) + clustering | 93.6 % | 9 |
| Buscador (confianza ≥ 0.65) + 1-NN | 95.4 % | 19 |
| Buscador (confianza ≥ 0.65) + clustering | 92.9 % | 19 |
| Buscador (confianza ≥ 0.75) + 1-NN | 95.7 % | 42 |
| Buscador (confianza ≥ 0.75) + clustering | 92.1 % | 42 |
| Buscador (confianza ≥ 0.85) + 1-NN | 95.4 % | 69 |
| Buscador (confianza ≥ 0.85) + clustering | 88.2 % | 69 |
| Buscador (confianza ≥ 0.95) + 1-NN | 94.6 % | 82 |
| Buscador (confianza ≥ 0.95) + clustering | 87.1 % | 82 |

Productos sin respuesta del buscador (9): ARROZ SOS, BEBER F SILVES 0%, CEBOLLA CONG ZIP, F. DEHESA 6L, FRESÓN, FUENTE DEHESA 1,5L, GRIEGO LIGERO NAT, LOTE 3 BAYETAS MICRO, T CHOCO CRUJIENTE
De ellos, el 1-NN acierta 7 y el clustering 4.

Nota: el umbral se elige con estos mismos datos (5 valores probados), así que la mejor cifra es algo optimista.
