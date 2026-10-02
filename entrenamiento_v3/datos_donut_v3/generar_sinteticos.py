"""
Generador de tickets sintéticos de Mercadona para ampliar el dataset de Donut.

Los tickets digitales de Mercadona tienen siempre la misma maqueta (fuente
Times/Nimbus Roman de 20,4 pt, columnas fijas, mismo logo...). Aprovechando
eso, se dibujan tickets nuevos con productos y precios aleatorios y se genera
su Ground Truth de forma exacta (se conoce porque lo hemos dibujado nosotros).

Fuentes de productos:
  - Nombres reales de los tickets del split de ENTRENAMIENTO (nunca validación,
    para no contaminar la evaluación).
  - Nombres del catálogo de Mercadona (catalogo_mercadona.csv) convertidos al
    estilo del ticket: MAYÚSCULAS, sin marca y cortados a 20 caracteres.

Uso:
    python generar_sinteticos.py <dataset_v3> <catalogo.csv|-> <n_tickets> [semilla]
Crea <dataset_v3>/synthetic/ con las imágenes y su metadata.jsonl.
"""
import csv
import json
import random
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 200
S = DPI / 72                   # puntos PDF -> píxeles
ANCHO_PAGINA = 612             # pt
TAM_FUENTE = 20.4              # pt
PASO = 25.5                    # pt entre líneas de producto
PASO_PESO = 22.4               # pt entre nombre de producto a peso y su línea de peso

FUENTES = {
    "regular": [
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "bold": [
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ],
}

MARCAS = {"HACENDADO", "DELIPLUS", "BOSQUE", "VERDE", "COMPY", "HIDA", "DANONE",
          "PASCUAL", "PULEVA", "NESTLÉ", "NESTLE", "COLGATE", "L'ORÉAL"}
FRUTA_VERDURA = ["PLATANO", "BANANA", "MANZANA GOLDEN", "PERA CONFERENCIA", "TOMATE CANARIO",
                 "TOMATE PERA", "PEPINO", "CALABACIN", "BERENJENA", "PIMIENTO ROJO", "PIMIENTO VERDE",
                 "CEBOLLA", "PATATA", "MANGO", "LIMON", "NARANJA ZUMO", "MANDARINA", "KIWI",
                 "MELOCOTON", "NECTARINA", "UVA BLANCA", "ZANAHORIA", "AGUACATE", "BROCOLI"]


def cargar_fuente(tipo: str, tam_px: int) -> ImageFont.FreeTypeFont:
    for ruta in FUENTES[tipo]:
        if Path(ruta).exists():
            return ImageFont.truetype(ruta, tam_px)
    raise FileNotFoundError("Instala fuentes: apt-get install fonts-urw-base35 fonts-liberation")


def euros(centimos: int) -> str:
    signo = "-" if centimos < 0 else ""
    c = abs(centimos)
    return f"{signo}{c // 100},{c % 100:02d}"


def nombre_estilo_ticket(nombre: str) -> str:
    palabras = [p for p in nombre.upper().split() if p not in MARCAS]
    texto = " ".join(palabras)
    return texto[:20].strip()


def cargar_vocabulario(dataset: Path, catalogo: str):
    nombres_reales, precios = [], []
    for linea in open(dataset / "train" / "metadata.jsonl", encoding="utf-8"):
        gt = json.loads(json.loads(linea)["ground_truth"])["gt_parse"]
        for it in gt["menu"]:
            if "weight" not in it and it["nm"] != "PARKING":
                nombres_reales.append(it["nm"])
                p = it.get("unitprice", it["price"])
                precios.append(int(round(float(p.replace(",", ".")) * 100)))
    nombres_catalogo = []
    if catalogo and catalogo != "-" and Path(catalogo).exists():
        with open(catalogo, encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f, delimiter=";"):
                n = nombre_estilo_ticket(fila["Nombre"])
                if len(n) >= 4:
                    nombres_catalogo.append(n)
                try:
                    precios.append(int(round(float(fila["Precio"]) * 100)))
                except ValueError:
                    pass
    return sorted(set(nombres_reales)), sorted(set(nombres_catalogo)), [p for p in precios if 10 <= p <= 3000]


class Generador:
    def __init__(self, dataset: Path, catalogo: str, semilla: int):
        self.rng = random.Random(semilla)
        self.reales, self.catalogo, self.precios = cargar_vocabulario(dataset, catalogo)
        assets = dataset / "assets"
        self.logo = Image.open(assets / "logo.png").convert("L")
        self.barras = Image.open(assets / "codigo_barras.png").convert("L")
        self.contactless = Image.open(assets / "contactless.png").convert("L")
        self.logo_tel = Image.open(assets / "logo_telefono.png").convert("L")
        self.tiendas = json.loads((assets / "tiendas.json").read_text(encoding="utf-8"))
        tam = round(TAM_FUENTE * S)
        self.f = cargar_fuente("regular", tam)
        self.fb = cargar_fuente("bold", tam)

    # ── Contenido aleatorio ──
    def nombre(self) -> str:
        # 60 % nombres reales de tickets, 40 % nombres del catálogo
        if self.catalogo and self.rng.random() < 0.4:
            return self.rng.choice(self.catalogo)
        return self.rng.choice(self.reales)

    def precio(self) -> int:
        return self.rng.choice(self.precios)

    def productos(self):
        r = self.rng.random()
        n = self.rng.randint(1, 3) if r < 0.3 else self.rng.randint(4, 12) if r < 0.8 else self.rng.randint(13, 34)
        items = []
        for _ in range(n):
            t = self.rng.random()
            if t < 0.08:
                kg = self.rng.randint(150, 2500)             # gramos
                eur_kg = self.rng.randint(79, 1299)          # céntimos/kg
                importe = round(kg * eur_kg / 1000)
                items.append({"cnt": "1", "nm": self.rng.choice(FRUTA_VERDURA),
                              "weight": f"{kg // 1000},{kg % 1000:03d} kg",
                              "unitprice": f"{euros(eur_kg)} €/kg", "price": euros(importe), "_c": importe})
            elif t < 0.22:
                cnt = self.rng.choice([2, 2, 2, 3, 3, 4, 5, 6])
                pu = self.precio()
                items.append({"cnt": str(cnt), "nm": self.nombre(), "unitprice": euros(pu),
                              "price": euros(pu * cnt), "_c": pu * cnt})
            else:
                pu = self.precio()
                items.append({"cnt": "1", "nm": self.nombre(), "price": euros(pu), "_c": pu})
        if self.rng.random() < 0.05:
            items.append({"cnt": "1", "nm": "PARKING", "price": "0,00", "_c": 0, "_parking": True})
        return items

    # ── Dibujo ──
    def texto(self, d, x_pt, y_pt, txt, bold=False, alinear="l"):
        fuente = self.fb if bold else self.f
        d.text((x_pt * S, y_pt * S), txt, font=fuente, fill=0, anchor=f"{alinear}a")

    def centrado(self, d, y_pt, txt, bold=False):
        self.texto(d, ANCHO_PAGINA / 2, y_pt, txt, bold, "m")

    def pegar(self, lienzo, img, x_pt, y_pt):
        lienzo.paste(img, (round(x_pt * S), round(y_pt * S)))

    def ticket(self):
        rng = self.rng
        devolucion = rng.random() < 0.03
        items = self.productos()
        if devolucion:
            it = items[0]
            c = -abs(it["_c"]) or -100
            items = [{"cnt": "-1", "nm": it["nm"], "price": euros(c), "_c": c}]
        total = sum(it["_c"] for it in items)

        fecha = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 1400), minutes=rng.randint(540, 1290))
        tienda = rng.choice(self.tiendas)

        alto = 460 + 26 * len(items) + 23 * sum("weight" in i for i in items) + 700
        lienzo = Image.new("L", (round(ANCHO_PAGINA * S), round(alto * S)), 255)
        d = ImageDraw.Draw(lienzo)

        # Cabecera
        self.pegar(lienzo, self.logo, 265, 38)
        y = 134
        self.centrado(d, y, "MERCADONA, S.A.   A-46103834", bold=True)
        if devolucion:
            y += 24.5
            self.centrado(d, y, "***** DEVOLUCIÓN *****")
        for linea in tienda["direccion"]:
            y += 24.5
            self.centrado(d, y, linea)
        y += 24.5
        self.centrado(d, y, f"TELÉFONO:   {tienda['telefono']}")
        y += 24.5
        self.centrado(d, y, f"{fecha:%d/%m/%Y %H:%M}      OP: {rng.randint(1000000, 4999999)}")
        y += 24.5
        fra = f"{rng.randint(1000, 4999)}-{rng.randint(1, 30):03d}-{rng.randint(1, 999999):06d}"
        self.centrado(d, y, f"{'DEVOLUCIÓN' if devolucion else 'FACTURA SIMPLIFICADA'}: {fra}")
        y += 31
        self.pegar(lienzo, self.barras, 51, y)
        y += 77

        # Tabla de productos
        self.texto(d, 83, y, "Descripción")
        self.texto(d, 461, y, "P. Unit", alinear="r")
        self.texto(d, 574.5, y, "Importe", alinear="r")
        d.line([(36 * S, (y + 22) * S), (576 * S, (y + 22) * S)], fill=0, width=max(1, round(0.8 * S)))
        y += PASO + 0.5
        for it in items:
            self.texto(d, 72.5, y, it["cnt"], alinear="r")
            self.texto(d, 82.7, y, it["nm"])
            if "weight" in it:
                y += PASO_PESO
                self.texto(d, 194.9, y, it["weight"], alinear="r")
                self.texto(d, 461.4, y, it["unitprice"], alinear="r")
                self.texto(d, 574.5, y, it["price"], alinear="r")
            else:
                if "unitprice" in it:
                    self.texto(d, 461.4, y, it["unitprice"], alinear="r")
                self.texto(d, 574.5, y, it["price"], alinear="r")
            if it.get("_parking"):
                y += PASO_PESO
                e = rng.randint(9, 20)
                self.texto(d, 145.3, y, f"ENTRADA  {e}:{rng.randint(10, 59)}       SALIDA  {e}:{rng.randint(10, 59)}")
            y += PASO + (0 if "weight" not in it else 0)

        # Totales
        y += 30
        self.texto(d, 491.5, y, "TOTAL (€)", bold=True, alinear="r")
        self.texto(d, 574.5, y, euros(total), bold=True, alinear="r")
        y += 22.4
        self.texto(d, 491.5, y, "TARJETA BANCARIA", bold=True, alinear="r")
        self.texto(d, 574.5, y, euros(total), bold=True, alinear="r")

        # Tabla de IVA (valores coherentes pero irrelevantes para el modelo)
        y += 47
        self.texto(d, 118, y, "IVA", alinear="m")
        self.texto(d, 306, y, "BASE IMPONIBLE (€)", alinear="m")
        self.texto(d, 494, y, "CUOTA (€)", alinear="m")
        restante, bases, cuotas = total, 0, 0
        tipos = sorted(rng.sample([4, 10, 21], rng.randint(1, 3)))
        for k, pct in enumerate(tipos):
            y += 24.5
            parte = restante if k == len(tipos) - 1 else round(restante * rng.uniform(0.2, 0.7))
            restante -= parte
            base = round(parte / (1 + pct / 100)); cuota = parte - base
            bases += base; cuotas += cuota
            self.texto(d, 118, y, f"{pct}%", alinear="m")
            self.texto(d, 306, y, euros(base), alinear="m")
            self.texto(d, 494, y, euros(cuota), alinear="m")
        y += 21.5
        self.texto(d, 118, y, "TOTAL", bold=True, alinear="m")
        self.texto(d, 306, y, euros(bases), bold=True, alinear="m")
        self.texto(d, 494, y, euros(cuotas), bold=True, alinear="m")

        # Pago
        y += 47
        tarjeta = f"{rng.randint(0, 9999):04d}"
        self.texto(d, 40.5, y, f"TARJ. BANCARIA:  **** **** **** {tarjeta}")
        y += 24
        self.texto(d, 40.5, y, f"N.C: {rng.randint(0, 99999999):09d}")
        self.texto(d, 573, y, f"AUT: {rng.randint(0, 999999):06d}", alinear="r")
        y += 24
        self.texto(d, 40.5, y, "AID: A0000000041010")
        self.texto(d, 573, y, "ARC: 00", alinear="r")
        y += 21
        self.pegar(lienzo, self.contactless, 286, y)
        y += 46
        red = rng.choice(["MASTERCARD", "VISA"])
        self.texto(d, 40.5, y, red)
        y += 24
        self.texto(d, 40.5, y, f"Importe: {euros(total)} €")
        self.texto(d, 573, y, red, alinear="r")
        y += 46
        self.pegar(lienzo, self.logo_tel, 205, y)
        y += 81
        self.centrado(d, y, "SE ADMITEN DEVOLUCIONES CON TICKET")
        lienzo = lienzo.crop((0, 0, lienzo.width, round((y + 60) * S)))

        menu = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
        gt = {"date": f"{fecha:%d/%m/%Y}", "menu": menu, "total": {"total_price": euros(total)}}
        return lienzo.convert("RGB"), gt


def main(dataset: Path, catalogo: str, n: int, semilla: int = 1234):
    gen = Generador(dataset, catalogo, semilla)
    salida = dataset / "synthetic"
    salida.mkdir(exist_ok=True)
    with open(salida / "metadata.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            img, gt = gen.ticket()
            nombre = f"sint_{i:05d}.jpg"
            img.save(salida / nombre, "JPEG", quality=90)
            f.write(json.dumps({"file_name": nombre,
                                "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False)},
                               ensure_ascii=False) + "\n")
    print(f"✅ {n} tickets sintéticos generados en {salida}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1234)
