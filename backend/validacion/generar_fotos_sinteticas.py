"""
Fotos SIMULADAS de tickets a partir de los PDF (prueba de robustez de Donut).

No hay fotos de tickets en papel con su PDF, así que se simula lo que estropea una foto de móvil sobre la
imagen del PDF: ticket sobre una mesa, girado y en perspectiva, iluminación desigual (sombra), desenfoque,
ruido del sensor, menos resolución y compresión JPEG. Tres niveles (leve, media, fuerte), con semilla
fija para que sea reproducible. Cada imagen se llama igual que su PDF, así evaluar_donut.py sabe cuál es
el ticket correcto.

Limitación (decirlo en la memoria): no reproduce el papel térmico arrugado o desvaído, los reflejos ni el
texto cortado; es una aproximación a una foto real, no un sustituto.

Uso (desde backend/):
    python -m validacion.generar_fotos_sinteticas              # PDF nuevos + los 14 de validación
    python -m validacion.generar_fotos_sinteticas --todos      # también los de entrenamiento
Después:
    python -m validacion.evaluar_donut --solo nuevo validacion --fotos validacion/fotos_sinteticas/leve validacion/fotos_sinteticas/media validacion/fotos_sinteticas/fuerte
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, ImageFilter

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))
from validacion.evaluar_donut import CARPETAS_PDF, gt_de_pdf, huella_gt, splits_entrenamiento  # noqa: E402

SALIDA = RAIZ / "fotos_sinteticas"
NIVELES = {
    #          giro º  perspectiva  desenfoque  ruido  luz   escala  calidad JPEG
    "leve":   dict(giro=2, persp=0.02, blur=0.6, ruido=4, luz=0.15, escala=0.85, jpeg=85),
    "media":  dict(giro=5, persp=0.05, blur=1.1, ruido=8, luz=0.30, escala=0.70, jpeg=70),
    "fuerte": dict(giro=9, persp=0.08, blur=1.7, ruido=12, luz=0.45, escala=0.55, jpeg=55),
}


def coeficientes_perspectiva(origen, destino):
    A, b = [], []
    for (x, y), (u, v) in zip(destino, origen):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def simular_foto(ticket: Image.Image, p: dict, rnd: random.Random) -> Image.Image:
    w, h = ticket.size
    # 1) Ticket sobre una mesa (fondo de color liso con algo de textura)
    margen = int(0.12 * max(w, h))
    fondo_color = rnd.choice([(150, 120, 90), (110, 110, 115), (190, 185, 175), (90, 70, 55)])
    lienzo = Image.new("RGB", (w + 2 * margen, h + 2 * margen), fondo_color)
    textura = np.random.default_rng(rnd.randint(0, 10 ** 6)).normal(0, 6, (lienzo.height, lienzo.width, 1))
    lienzo = Image.fromarray(np.clip(np.asarray(lienzo, float) + textura, 0, 255).astype(np.uint8))
    lienzo.paste(ticket, (margen, margen))

    # 2) Giro + perspectiva (el móvil no está paralelo al ticket)
    angulo = rnd.uniform(-p["giro"], p["giro"])
    lienzo = lienzo.rotate(angulo, resample=Image.BICUBIC, expand=False, fillcolor=fondo_color)
    W, H = lienzo.size
    d = p["persp"]
    esquinas = [(0, 0), (W, 0), (W, H), (0, H)]
    desplazadas = [(x + rnd.uniform(-d, d) * W, y + rnd.uniform(-d, d) * H) for x, y in esquinas]
    lienzo = lienzo.transform((W, H), Image.PERSPECTIVE, coeficientes_perspectiva(esquinas, desplazadas),
                              resample=Image.BICUBIC, fillcolor=fondo_color)

    # 3) Iluminación desigual: degradado de sombra en una dirección al azar
    arr = np.asarray(lienzo, float)
    yy, xx = np.mgrid[0:H, 0:W]
    ang = rnd.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    grad = (grad - grad.min()) / max(1e-9, grad.max() - grad.min())
    arr *= (1 - p["luz"] * grad)[..., None]
    arr *= rnd.uniform(0.85, 1.05)                                   # exposición
    arr[..., 2] *= rnd.uniform(0.9, 1.0)                             # luz cálida
    # 4) Ruido del sensor
    arr += np.random.default_rng(rnd.randint(0, 10 ** 6)).normal(0, p["ruido"], arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # 5) Desenfoque (enfoque imperfecto / pulso) y menos resolución
    img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    img = img.resize((int(W * p["escala"]), int(H * p["escala"])), Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--todos", action="store_true", help="incluye los tickets de entrenamiento")
    args = ap.parse_args()
    splits = splits_entrenamiento()
    hechos, vistos = 0, set()
    for carpeta in CARPETAS_PDF:
        if not carpeta.exists():
            continue
        for pdf in sorted(carpeta.glob("*.pdf"), key=lambda p: ("(" in p.stem, p.name)):
            try:
                gt = gt_de_pdf(pdf)
            except Exception:
                continue
            huella = huella_gt(gt)
            grupo = splits.get(huella) or ("nuevo" if carpeta.name == "tickets_nuevos" else "desconocido")
            if huella in vistos or (grupo == "train" and not args.todos):
                continue
            vistos.add(huella)
            with pdfplumber.open(pdf) as doc:
                ticket = doc.pages[0].to_image(resolution=200).original.convert("RGB")
            semilla = int(hashlib.md5(pdf.name.encode()).hexdigest()[:8], 16)
            for nivel, p in NIVELES.items():
                destino = SALIDA / nivel
                destino.mkdir(parents=True, exist_ok=True)
                simular_foto(ticket, p, random.Random(semilla + len(nivel))).save(
                    destino / f"{pdf.stem}.jpg", "JPEG", quality=p["jpeg"])
            hechos += 1
            print(f"  📷 {pdf.name} ({grupo})")
    print(f"✅ {hechos} tickets × {len(NIVELES)} niveles en {SALIDA}")


if __name__ == "__main__":
    main()
