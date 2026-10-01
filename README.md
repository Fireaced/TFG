# FinTrack Pro — Analizador de tickets con IA

Aplicación web de código abierto para digitalizar tickets de compra de Mercadona y llevar el control de los gastos personales. Es el Trabajo de Fin de Grado en Ingeniería de Computadores.

## Cómo funciona

1. El usuario sube un ticket (PDF o imagen).
2. Un modelo **Donut** (*Document Understanding Transformer*) ajustado a los tickets de Mercadona extrae la fecha, los productos, las cantidades, los precios y el total.
3. El usuario revisa los datos: puede corregirlos, añadir productos que falten o quitar los que sobren.
4. Cada producto se clasifica automáticamente en una o varias categorías a partir del catálogo de Mercadona (`backend/catalogo_mercadona.csv`). Si el usuario corrige una categoría, el sistema la recuerda para los próximos tickets.
5. El ticket se guarda en MongoDB y se puede consultar y filtrar por categoría.

## Estructura

| Carpeta | Contenido |
|---|---|
| `backend/` | API en FastAPI (`server.py`), categorizador (`categorizador.py`) y scraper del catálogo (`Scrapper.py`) |
| `backend/dataset-mercadona/` | Tickets originales y scripts para generar el dataset (`preparar_dataset.py`, `generar_sinteticos.py`) |
| `entrenamiento_v3/` | Cuaderno de Google Colab para entrenar el modelo |
| `frontend/` | Interfaz web en React + Tailwind |
| `Memoria/` | Memoria del TFG en LaTeX |

## Puesta en marcha

**Backend**

```bash
cd backend
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```

Hace falta MongoDB en local y un fichero `backend/.env` con `MONGO_URL` y `DB_NAME`. El modelo entrenado se descomprime en `backend/mi_modelo_mercadona_v3/`.

**Frontend**

```bash
cd frontend
yarn install
yarn start
```

Necesita un fichero `frontend/.env` con `REACT_APP_BACKEND_URL=http://localhost:8001`.
