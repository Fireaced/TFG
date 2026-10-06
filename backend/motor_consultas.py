"""
Motor de consultas sobre los gastos.

Toda pregunta (venga del LLM o del motor de reglas) se traduce a una `Consulta`
estructurada que combina piezas independientes:

    medida      qué se mide          gasto | unidades | compras | precio
    operacion   qué se calcula       total | media | maximo | minimo | ranking | evolucion | ultimo | lista
    agrupar_por por qué se agrupa    producto | categoria | subcategoria | ticket | mes | semana | dia_semana
    por         unidad de la media   compra | dia | semana | mes
    filtros     qué datos entran     concepto, categoría, periodo, devoluciones, día de la semana, importe…

Este módulo ejecuta la consulta con pandas y redacta la respuesta. Los números
los calcula siempre este código, nunca el modelo de lenguaje.
"""
from datetime import date, timedelta
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


# ════════════════════════════════════════════════════════════════════════
# Esquema de la consulta
# ════════════════════════════════════════════════════════════════════════
class Filtros(BaseModel):
    concepto: Optional[str] = Field(None, description="Producto o tipo de producto con las palabras del usuario: 'lácteos', 'yogures', 'pan de molde'")
    categoria: Optional[str] = Field(None, description="Nombre exacto de una categoría del catálogo, si el usuario la nombra")
    periodo: Optional[str] = Field(None, description="Expresión temporal tal cual: 'este mes', 'el mes pasado', 'en marzo de 2025', 'últimos 3 meses'")
    desde: Optional[date] = None
    hasta: Optional[date] = None
    tipo_ticket: Literal["todos", "compras", "devoluciones"] = "todos"
    dias_semana: List[int] = Field(default_factory=list, description="0=lunes … 6=domingo")
    importe_min: Optional[float] = Field(None, description="Importe mínimo del ticket en euros")
    importe_max: Optional[float] = Field(None, description="Importe máximo del ticket en euros")

    @field_validator("dias_semana")
    @classmethod
    def _dias_validos(cls, v):
        return [d for d in v if 0 <= d <= 6]


class Consulta(BaseModel):
    tipo: Literal["consulta", "saludo", "fuera_de_ambito"] = "consulta"
    medida: Literal["gasto", "unidades", "compras", "precio"] = "gasto"
    operacion: Literal["total", "media", "maximo", "minimo", "ranking", "evolucion", "ultimo", "lista"] = "total"
    agrupar_por: Optional[Literal["producto", "categoria", "subcategoria", "ticket", "mes", "semana", "dia_semana"]] = None
    por: Optional[Literal["compra", "dia", "semana", "mes"]] = None
    orden: Literal["desc", "asc"] = "desc"
    limite: int = Field(5, ge=1, le=50)
    filtros: Filtros = Field(default_factory=Filtros)


# ════════════════════════════════════════════════════════════════════════
# Utilidades de formato
# ════════════════════════════════════════════════════════════════════════
def euros(v: float) -> str:
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def num(v: float, dec: int = 1) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{dec}f}".replace(".", ",")


def fecha_es(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def etiqueta_periodo(ini: Optional[date], fin: Optional[date]) -> str:
    if not ini and not fin:
        return "en todo tu historial"
    ini = ini or date(2000, 1, 1)
    fin = fin or date.today()
    if ini.day == 1 and (fin + timedelta(days=1)).day == 1 and ini.year == fin.year and ini.month == fin.month:
        return f"en {MESES[ini.month - 1]} de {ini.year}"
    if ini == date(ini.year, 1, 1) and fin == date(ini.year, 12, 31):
        return f"en {ini.year}"
    if ini == fin:
        return f"el {fecha_es(ini)}"
    return f"del {fecha_es(ini)} al {fecha_es(fin)}"


def _tabla(columnas, filas):
    return {"columnas": columnas, "filas": filas} if filas else None


def _plural(n, singular, plural=None):
    return singular if n == 1 else (plural or singular + "s")


# ════════════════════════════════════════════════════════════════════════
# Ejecución
# ════════════════════════════════════════════════════════════════════════
class EjecutorConsultas:
    def __init__(self, resolutor, clusters, arbol_categorias, parsear_periodo):
        self.resolutor = resolutor
        self.clusters = clusters
        self.categorias = [g["categoria"] for g in arbol_categorias]
        self.parsear_periodo = parsear_periodo   # función (texto, hoy) -> (ini, fin, etiqueta, resto)

    # ── Filtros ──
    def _periodo(self, f: Filtros, hoy: date):
        if f.desde or f.hasta:
            return f.desde, f.hasta, etiqueta_periodo(f.desde, f.hasta)
        if f.periodo:
            ini, fin, etiqueta, _ = self.parsear_periodo(f.periodo, hoy)
            return ini, fin, etiqueta
        return None, None, "en todo tu historial"

    def _categoria_exacta(self, nombre: str) -> Optional[str]:
        from asistente_gastos import sin_tildes
        n = sin_tildes(nombre).strip()
        for c in self.categorias:
            if sin_tildes(c) == n or sin_tildes(c).startswith(n):
                return c
        return None

    def ejecutar(self, c: Consulta, df: pd.DataFrame, hoy: date) -> dict:
        if c.tipo == "saludo":
            return {"respuesta": "Estoy aquí para ayudarte con tus gastos. Puedes preguntarme, por ejemplo:", "sugerencias": SUGERENCIAS}
        if c.tipo == "fuera_de_ambito":
            return {"respuesta": "Solo puedo responder preguntas sobre tus tickets y tus gastos. Prueba, por ejemplo, con:",
                    "sugerencias": SUGERENCIAS[:4]}
        if not len(df):
            return {"respuesta": "Todavía no tienes tickets guardados. Sube alguno y pregúntame lo que quieras."}

        f = c.filtros
        ini, fin, per = self._periodo(f, hoy)
        d = df
        if ini:
            d = d[d["fecha"] >= ini]
        if fin:
            d = d[d["fecha"] <= fin]
        if f.tipo_ticket == "devoluciones":
            d = d[d["es_devolucion"]]
        elif f.tipo_ticket == "compras" or not (c.operacion in ("total", "evolucion") and c.medida in ("gasto", "unidades")):
            # Las devoluciones solo restan en los totales; en medias, máximos, rankings, listados
            # o al contar compras se dejan fuera para no falsear el resultado.
            d = d[~d["es_devolucion"]]
        if c.medida in ("gasto", "precio") and (c.orden == "asc" or c.operacion == "minimo") and f.tipo_ticket != "devoluciones":
            d = d[d["importe"] > 0]   # líneas a 0 € (p. ej. PARKING) no cuentan como "lo más barato"
        if f.dias_semana:
            d = d[d["fecha"].apply(lambda x: x.weekday()).isin(f.dias_semana)]
        if f.importe_min is not None:
            d = d[d["total_ticket"] >= f.importe_min]
        if f.importe_max is not None:
            d = d[d["total_ticket"] <= f.importe_max]

        extras = []
        if f.tipo_ticket == "devoluciones":
            extras.append("solo devoluciones")
        if f.dias_semana:
            extras.append("solo los " + ", ".join(DIAS[x] + ("s" if x >= 5 else "") for x in f.dias_semana))
        if f.importe_min is not None:
            extras.append(f"tickets de más de {euros(f.importe_min)}")
        if f.importe_max is not None:
            extras.append(f"tickets de menos de {euros(f.importe_max)}")
        if extras:
            per = f"{per} ({'; '.join(extras)})"
        concepto, detalle = "", {}
        if f.categoria:
            cat = self._categoria_exacta(f.categoria)
            if cat:
                d = d[d["categorias_lista"].apply(lambda cs: any(x[0] == cat for x in cs))]
                concepto = cat
            elif not f.concepto:
                f.concepto = f.categoria
        if f.concepto:
            from asistente_gastos import sin_tildes
            mascara, etiqueta, detalle = self.resolutor.resolver(sin_tildes(f.concepto), d, self.clusters)
            if mascara is None:
                return {"respuesta": f"No he encontrado ningún producto que corresponda a «{f.concepto}» {per}.",
                        "contexto": {"periodo": per}}
            d = d[mascara]
            concepto = etiqueta if etiqueta != sin_tildes(f.concepto) else f.concepto

        ctx = Contexto(c, d, df, per, concepto, ini, fin)
        if not len(d):
            que = "devoluciones" if f.tipo_ticket == "devoluciones" else "compras"
            return {"respuesta": f"No hay {que} que coincidan{(' con ' + concepto) if concepto else ''} {per}.".replace("  ", " ")}

        resultado = getattr(self, f"_op_{c.operacion}")(ctx)
        if detalle.get("tipo") == "productos" and "nota" not in resultado:
            prods = detalle["productos"]
            resultado["nota"] = "Productos incluidos: " + ", ".join(prods[:12]) + ("…" if len(prods) > 12 else "")
        if ini and df["fecha"].max() < ini:
            resultado["nota"] = f"Tu último ticket guardado es del {fecha_es(df['fecha'].max())}."
        return resultado

    # ── Agrupaciones ──
    @staticmethod
    def _claves(d: pd.DataFrame, agrupar: str) -> pd.DataFrame:
        if agrupar == "producto":
            return d.assign(clave=d["descripcion"])
        if agrupar in ("categoria", "subcategoria"):
            i = 0 if agrupar == "categoria" else 1
            filas = d.assign(clave=d["categorias_lista"].apply(lambda cs: sorted({(x[i] or x[0]) for x in cs}) or ["Sin categoría"]))
            return filas.explode("clave")
        if agrupar == "ticket":
            return d.assign(clave=d["ticket_id"])
        if agrupar == "mes":
            return d.assign(clave=d["fecha"].apply(lambda x: f"{x.year}-{x.month:02d}"))
        if agrupar == "semana":
            return d.assign(clave=d["fecha"].apply(lambda x: (x - timedelta(days=x.weekday())).isoformat()))
        if agrupar == "dia_semana":
            return d.assign(clave=d["fecha"].apply(lambda x: x.weekday()))
        raise ValueError(agrupar)

    @staticmethod
    def _agregar(g: pd.DataFrame) -> pd.DataFrame:
        return g.groupby("clave").agg(gasto=("importe", "sum"), unidades=("cantidad", "sum"),
                                      compras=("ticket_id", "nunique"), precio=("precio_unitario", "mean"),
                                      fecha=("fecha", "max"))

    @staticmethod
    def _nombre_clave(clave, agrupar, d):
        if agrupar == "mes":
            return f"{MESES[int(clave[5:]) - 1].capitalize()} {clave[:4]}"
        if agrupar == "semana":
            return f"Semana del {fecha_es(date.fromisoformat(clave))}"
        if agrupar == "dia_semana":
            return DIAS[int(clave)].capitalize()
        if agrupar == "ticket":
            fila = d[d["ticket_id"] == clave].iloc[0]
            return f"{fecha_es(fila['fecha'])} · {fila['nombre_ticket']}"
        return str(clave)

    def _valor_txt(self, medida, v):
        return {"gasto": euros(v), "precio": euros(v), "unidades": f"{num(v)} {_plural(v, 'unidad', 'unidades')}",
                "compras": f"{num(v)} {_plural(v, 'compra')}"}[medida]

    # ── Operaciones ──
    def _op_total(self, x: "Contexto"):
        d, c = x.d, x.c
        n_tk = d["ticket_id"].nunique()
        dev = c.filtros.tipo_ticket == "devoluciones"
        if c.medida == "precio":
            return self._op_media(x)
        if c.medida == "gasto":
            total = d["importe"].sum()
            if dev:
                txt = f"Te han devuelto **{euros(abs(total))}** {x.frase} en {n_tk} {_plural(n_tk, 'devolución', 'devoluciones')}."
            else:
                txt = f"Has gastado **{euros(total)}** {x.frase}, en {n_tk} {_plural(n_tk, 'compra')}."
        elif c.medida == "unidades":
            u = d["cantidad"].sum()
            txt = f"Has comprado **{num(u)} {_plural(u, 'unidad', 'unidades')}**{(' de ' + x.concepto) if x.concepto else ''} {x.per}, en {n_tk} {_plural(n_tk, 'compra')}."
        else:
            que = _plural(n_tk, "devolución", "devoluciones") if dev else _plural(n_tk, "compra")
            total = d.drop_duplicates("ticket_id")["total_ticket"].sum()
            txt = f"Has hecho **{n_tk} {que}** {x.frase}" + (f", por un total de {euros(abs(total))}." if not x.concepto else ".")
        tabla = None
        if dev or (c.medida == "compras" and n_tk <= 15):
            tabla = self._tabla_tickets(d, 15)
        elif x.concepto:
            tabla = self._tabla_productos(d, 8)
        return {"respuesta": txt, "tabla": tabla}

    def _op_media(self, x: "Contexto"):
        d, c = x.d, x.c
        if c.medida == "precio":
            pm = d["precio_unitario"].mean()
            txt = f"El precio medio{(' de ' + x.concepto) if x.concepto else ''} {x.per} es **{euros(pm)}**."
            return {"respuesta": txt, "tabla": self._tabla_productos(d, 8, precio=True)}
        por = c.por or ("semana" if c.medida == "compras" else "compra")
        if por == "compra":
            por_tk = d.groupby("ticket_id").agg(gasto=("importe", "sum"), unidades=("cantidad", "sum"))
            if c.medida == "compras":
                por = "semana"
            else:
                v = por_tk[c.medida].mean()
                txt = (f"Gastas de media **{euros(v)} por compra** {x.frase} ({len(por_tk)} compras)." if c.medida == "gasto" else
                       f"Compras de media **{num(v)} {_plural(v, 'unidad', 'unidades')}{(' de ' + x.concepto) if x.concepto else ''} por compra** {x.per}.")
                return {"respuesta": txt}
        dias = x.dias_rango()
        divisor = {"dia": 1, "semana": 7, "mes": 30.44}[por] if por != "compra" else 7
        n_periodos = max(dias / divisor, 1e-9)
        etiqueta = {"dia": "al día", "semana": "a la semana", "mes": "al mes"}[por]
        if c.medida == "gasto":
            txt = f"Gastas de media **{euros(d['importe'].sum() / n_periodos)} {etiqueta}** {x.frase}."
        elif c.medida == "unidades":
            v = d["cantidad"].sum() / n_periodos
            txt = f"Compras de media **{num(v)} {_plural(v, 'unidad', 'unidades')}{(' de ' + x.concepto) if x.concepto else ''} {etiqueta}** {x.per}."
        else:
            fechas = sorted(d.groupby("ticket_id")["fecha"].first())
            veces = len(fechas)
            v = veces / n_periodos
            sujeto = f"Compras {x.concepto}" if x.concepto else "Vas a comprar"
            txt = f"{sujeto} unas **{num(v)} {_plural(v, 'vez', 'veces')} {etiqueta}** {x.per} ({veces} {_plural(veces, 'vez', 'veces')} en total)."
            if veces > 1:
                media_dias = (pd.Timestamp(fechas[-1]) - pd.Timestamp(fechas[0])).days / (veces - 1)
                txt += f" De media pasan {num(media_dias, 0)} días entre una y otra; la última fue el {fecha_es(fechas[-1])}."
        return {"respuesta": txt}

    def _op_maximo(self, x: "Contexto"):
        return self._extremo(x, mayor=True)

    def _op_minimo(self, x: "Contexto"):
        return self._extremo(x, mayor=False)

    def _extremo(self, x: "Contexto", mayor: bool):
        d, c = x.d, x.c
        agrupar = c.agrupar_por or ("producto" if c.medida == "precio" else "ticket")
        adj = "más" if mayor else "menos"
        if c.medida == "precio" and agrupar == "producto":
            fila = d.sort_values("precio_unitario", ascending=not mayor).iloc[0]
            txt = (f"El producto {'más caro' if mayor else 'más barato'}{(' de ' + x.concepto) if x.concepto else ''} {x.per} es "
                   f"**{fila['descripcion']}**, a **{euros(fila['precio_unitario'])}** (el {fecha_es(fila['fecha'])}).")
            top = d.sort_values("precio_unitario", ascending=not mayor).drop_duplicates("descripcion").head(c.limite if c.limite > 1 else 8)
            return {"respuesta": txt, "tabla": _tabla(["Producto", "Precio", "Fecha"],
                                                     [[r.descripcion, euros(r.precio_unitario), fecha_es(r.fecha)] for r in top.itertuples()])}
        if agrupar == "ticket":
            tk = d.groupby("ticket_id").agg(fecha=("fecha", "first"), total=("total_ticket", "first"),
                                            parcial=("importe", "sum"), productos=("cantidad", "sum"))
            col = "parcial" if x.concepto else "total"
            if c.filtros.tipo_ticket != "devoluciones":
                tk = tk[tk[col] > 0] if len(tk[tk[col] > 0]) else tk
            fila_id = (tk[col].idxmax() if mayor else tk[col].idxmin())
            fila = tk.loc[fila_id]
            quien = "devolución" if c.filtros.tipo_ticket == "devoluciones" else "compra"
            if x.concepto:
                txt = (f"La {quien} en la que {adj} gastaste en {x.concepto} {x.per} fue la del **{fecha_es(fila.fecha)}**: "
                       f"**{euros(fila.parcial)}** (total del ticket {euros(fila.total)}).")
            else:
                txt = (f"Tu {quien} {'más cara' if mayor else 'más barata'} {x.per} fue la del **{fecha_es(fila.fecha)}**: "
                       f"**{euros(fila.total)}** con {num(fila.productos)} {_plural(fila.productos, 'producto')}.")
            prods = x.d[x.d["ticket_id"] == fila_id].sort_values("importe", ascending=False)
            tabla = _tabla(["Producto", "Cant.", "Importe"], [[r.descripcion, num(r.cantidad), euros(r.importe)] for r in prods.head(12).itertuples()])
            return {"respuesta": txt, "tabla": tabla, "ticket_id": fila_id}
        agg = self._agregar(self._claves(d, agrupar)).sort_values(c.medida, ascending=not mayor)
        clave, fila = agg.index[0], agg.iloc[0]
        nombre = self._nombre_clave(clave, agrupar, d)
        txt = f"**{nombre}** es {self._articulo(agrupar)} con {adj} {self._nombre_medida(c.medida)}{(' de ' + x.concepto) if x.concepto else ''} {x.per}: **{self._valor_txt(c.medida, fila[c.medida])}**."
        return {"respuesta": txt, "tabla": self._tabla_ranking(agg.head(8), agrupar, d)}

    def _op_ranking(self, x: "Contexto"):
        d, c = x.d, x.c
        agrupar = c.agrupar_por or "producto"
        agg = self._agregar(self._claves(d, agrupar))
        agg = agg.drop(index="Sin categoría", errors="ignore") if agrupar in ("categoria", "subcategoria") else agg
        orden = [c.medida, "compras"] if c.medida == "unidades" else [c.medida]
        agg = agg.sort_values(orden, ascending=c.orden == "asc")
        if agrupar in ("mes", "semana") and c.orden == "desc" and c.limite >= 12:
            agg = agg.sort_index()
        clave, fila = agg.index[0], agg.iloc[0]
        nombre = self._nombre_clave(clave, agrupar, d)
        adj = "más" if c.orden == "desc" else "menos"
        txt = (f"{self._articulo(agrupar, mayus=True)} con {adj} {self._nombre_medida(c.medida)}{(' de ' + x.concepto) if x.concepto else ''} {x.per} "
               f"es **{nombre}**: **{self._valor_txt(c.medida, fila[c.medida])}**.")
        return {"respuesta": txt, "tabla": self._tabla_ranking(agg.head(c.limite if c.limite > 1 else 5), agrupar, d)}

    def _op_evolucion(self, x: "Contexto"):
        d, c = x.d, x.c
        if c.medida == "precio":
            filas = []
            for prod, g in d.sort_values("fecha").groupby("descripcion"):
                p0, p1 = g.iloc[0], g.iloc[-1]
                var = (p1.precio_unitario - p0.precio_unitario) / p0.precio_unitario * 100 if p0.precio_unitario else 0
                filas.append([prod, euros(p0.precio_unitario), fecha_es(p0.fecha), euros(p1.precio_unitario), fecha_es(p1.fecha),
                              f"{var:+.0f} %" if len(g) > 1 else "—"])
            if len(filas) == 1:
                f = filas[0]
                txt = (f"Has pagado **{f[0]}** a {f[1]} ({f[2]}) y la última vez a **{f[3]}** ({f[4]}): variación **{f[5]}**."
                       if f[5] != "—" else f"Solo tienes una compra de {f[0]}: **{f[3]}** ({f[4]}).")
            else:
                txt = f"Esta es la evolución del precio de {x.concepto or 'tus productos'} {x.per} (primer y último precio pagado)."
            return {"respuesta": txt, "tabla": _tabla(["Producto", "Primer precio", "Fecha", "Último precio", "Fecha", "Variación"], filas[:12])}
        agrupar = c.agrupar_por if c.agrupar_por in ("mes", "semana", "dia_semana") else "mes"
        agg = self._agregar(self._claves(d, agrupar)).sort_index()
        filas = [[self._nombre_clave(k, agrupar, d), num(r.compras), euros(r.gasto)] for k, r in agg.tail(24).iterrows()]
        txt = f"Este es tu {self._nombre_medida(c.medida)}{(' en ' + x.concepto) if x.concepto else ''} por {'meses' if agrupar == 'mes' else 'semanas' if agrupar == 'semana' else 'día de la semana'} {x.per}."
        if agrupar != "dia_semana" and len(agg) >= 2:
            a, b = agg.iloc[-1], agg.iloc[-2]
            dif = a[c.medida] - b[c.medida]
            txt += (f" En el último ({self._nombre_clave(agg.index[-1], agrupar, d).lower()}) fue {self._valor_txt(c.medida, a[c.medida])}, "
                    f"{'más' if dif > 0 else 'menos'} que en el anterior ({self._valor_txt(c.medida, abs(dif))} de diferencia).")
        return {"respuesta": txt, "tabla": _tabla(["Periodo", "Compras", "Gasto"], filas)}

    def _op_ultimo(self, x: "Contexto"):
        d = x.d.sort_values("fecha")
        fila = d.iloc[-1]
        if x.concepto:
            txt = f"La última vez que compraste {x.concepto} fue el **{fecha_es(fila['fecha'])}** ({fila['descripcion']}, {euros(fila['importe'])})."
        elif x.c.filtros.tipo_ticket == "devoluciones":
            txt = f"Tu última devolución fue el **{fecha_es(fila['fecha'])}**, por {euros(abs(fila['total_ticket']))}."
        else:
            txt = f"Tu última compra {x.per if x.per != 'en todo tu historial' else ''} fue el **{fecha_es(fila['fecha'])}**, por {euros(fila['total_ticket'])}.".replace("  ", " ")
        return {"respuesta": txt}

    def _op_lista(self, x: "Contexto"):
        d, c = x.d, x.c
        n = d["ticket_id"].nunique()
        dev = c.filtros.tipo_ticket == "devoluciones"
        if c.agrupar_por == "producto" or (x.concepto and c.agrupar_por != "ticket"):
            prods = d["descripcion"].nunique()
            return {"respuesta": f"He encontrado **{prods} {_plural(prods, 'producto')}** distintos{(' de ' + x.concepto) if x.concepto else ''} {x.per}.",
                    "tabla": self._tabla_productos(d, 20)}
        que = _plural(n, "devolución", "devoluciones") if dev else _plural(n, "ticket")
        return {"respuesta": f"He encontrado **{n} {que}** {x.frase}.", "tabla": self._tabla_tickets(d, 20)}

    # ── Tablas y textos auxiliares ──
    @staticmethod
    def _tabla_tickets(d, limite):
        tk = d.groupby("ticket_id").agg(fecha=("fecha", "first"), nombre=("nombre_ticket", "first"),
                                        total=("total_ticket", "first"), productos=("cantidad", "sum")).sort_values("fecha", ascending=False)
        return _tabla(["Fecha", "Ticket", "Productos", "Total"],
                      [[fecha_es(r.fecha), r.nombre, num(r.productos), euros(r.total)] for r in tk.head(limite).itertuples()])

    @staticmethod
    def _tabla_productos(d, limite, precio=False):
        agg = d.groupby("descripcion").agg(unidades=("cantidad", "sum"), compras=("ticket_id", "nunique"),
                                           gasto=("importe", "sum"), precio=("precio_unitario", "mean")).sort_values("gasto", ascending=False)
        if precio:
            return _tabla(["Producto", "Precio medio", "Compras"], [[k, euros(r.precio), num(r.compras)] for k, r in agg.head(limite).iterrows()])
        return _tabla(["Producto", "Unidades", "Gasto"], [[k, num(r.unidades), euros(r.gasto)] for k, r in agg.head(limite).iterrows()])

    def _tabla_ranking(self, agg, agrupar, d):
        cab = {"producto": "Producto", "categoria": "Categoría", "subcategoria": "Subcategoría", "ticket": "Ticket",
               "mes": "Mes", "semana": "Semana", "dia_semana": "Día"}[agrupar]
        return _tabla([cab, "Unidades", "Compras", "Gasto"],
                      [[self._nombre_clave(k, agrupar, d), num(r.unidades), num(r.compras), euros(r.gasto)] for k, r in agg.iterrows()])

    @staticmethod
    def _nombre_medida(m):
        return {"gasto": "gasto", "unidades": "unidades compradas", "compras": "compras", "precio": "precio"}[m]

    @staticmethod
    def _articulo(agrupar, mayus=False):
        t = {"producto": "el producto", "categoria": "la categoría", "subcategoria": "la subcategoría", "ticket": "el ticket",
             "mes": "el mes", "semana": "la semana", "dia_semana": "el día de la semana"}[agrupar]
        return t[0].upper() + t[1:] if mayus else t


class Contexto:
    def __init__(self, c, d, df, per, concepto, ini, fin):
        self.c, self.d, self.df, self.per, self.concepto, self.ini, self.fin = c, d, df, per, concepto, ini, fin

    @property
    def frase(self):
        per = self.per
        if self.c.filtros.tipo_ticket == "devoluciones" and self.c.medida in ("gasto", "compras"):
            per = per.replace(" (solo devoluciones)", "").replace("solo devoluciones; ", "")
        return f"en {self.concepto} {per}" if self.concepto else per

    def dias_rango(self):
        """Días del periodo consultado (recortado a las fechas en las que hay tickets)."""
        inicio = max(self.ini, self.df["fecha"].min()) if self.ini else self.df["fecha"].min()
        final = min(self.fin, self.df["fecha"].max()) if self.fin else self.df["fecha"].max()
        return max(1, (final - inicio).days + 1)


def explicar(c: Consulta, per: str = "") -> str:
    """Descripción breve de cómo se ha interpretado la pregunta (se muestra bajo la respuesta)."""
    if c.tipo != "consulta":
        return ""
    partes = [{"gasto": "gasto", "unidades": "unidades", "compras": "nº de compras", "precio": "precio"}[c.medida],
              {"total": "total", "media": "media", "maximo": "máximo", "minimo": "mínimo", "ranking": "ranking",
               "evolucion": "evolución", "ultimo": "última vez", "lista": "listado"}[c.operacion]]
    if c.por and c.operacion == "media":
        partes[-1] += f" por {c.por}"
    if c.agrupar_por:
        partes.append(f"por {c.agrupar_por.replace('_', ' ')}")
    f = c.filtros
    if f.categoria:
        partes.append(f"categoría «{f.categoria}»")
    if f.concepto:
        partes.append(f"«{f.concepto}»")
    if f.tipo_ticket != "todos":
        partes.append(f.tipo_ticket)
    if f.dias_semana:
        partes.append("los " + ", ".join(DIAS[d] + ("s" if d >= 5 else "") for d in f.dias_semana))
    if f.importe_min is not None:
        partes.append(f"tickets ≥ {euros(f.importe_min)}")
    if f.importe_max is not None:
        partes.append(f"tickets ≤ {euros(f.importe_max)}")
    if per:
        partes.append(per)
    return " · ".join(partes)


SUGERENCIAS = [
    "¿Cuánto he gastado en lácteos este mes?",
    "¿Cuántas veces compro yogures a la semana?",
    "¿Cuál ha sido mi compra más cara del mes?",
    "¿De qué categoría compro más productos?",
    "¿Cuál es el producto que más he comprado?",
    "¿Qué día de la semana gasto más?",
    "¿Cuántas devoluciones he hecho?",
]
