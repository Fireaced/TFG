import requests
import pandas as pd
import time

# Mapa plano: ID -> Nombre específico de la sección
SUBCATEGORIAS = {
    112: "ACEITES", 115: "ESPECIAS", 116: "Mayonesa, ketchup y mostaza", 117: "Otras salsas",
    156: "Agua", 163: "Isotonico", 158: "Cola", 159: "Refresco de naranja y de limón", 161: "Tonica y Bitter", 162: "Refresco de te y sin gas",
    135: "Aceitunas y encurtidos", 133: "Frutos secos y fruta desecada", 132: "Patatas fritas y snacks",
    118: "Arroz", 121: "Legumbres", 120: "Pasta y fideos",
    89: "Azucar y edulcorante", 95: "Chicles y caramelos", 92: "Chocolate", 97: "Golosinas", 90: "Mermelada y miel",
    216: "Alimentación infantil", 219: "Biberón y chupete", 218: "Higiene y cuidado", 217: "Toallitas y pañales",
    164: "Cerveza", 166: "Cerveza sin alcohol", 181: "Licores", 174: "Sidra y cava", 168: "Tinto de verano y sangría", 170: "Vino blanco", 173: "Vino lambrusco y espumoso", 171: "Vino rosado", 169: "Vino tinto",
    86: "Cacao soluble y chocolate a la taza", 81: "Café cápsula y monodosis", 83: "Café molido y en grano", 84: "Café soluble y otras bebidas", 88: "Té e infusiones",
    46: "Arreglos", 38: "Aves y pollo", 47: "Carne congelada", 37: "Cerdo", 42: "Conejo y cordero", 43: "Embutido", 44: "Hamburguesas y picadas", 40: "Vacuno", 45: "Empanados y elaborados",
    78: "Cereales", 80: "Galletas", 79: "Tortitas",
    48: "Aves y jamón cocido", 52: "Bacón y salchichas", 49: "Chopped y mortadela", 51: "Embutido curado", 50: "Jamón serrano", 58: "Paté y sobrasada", 54: "Queso curado, semicurado y tierno", 56: "Queso lonchas, rallado y en porciones", 53: "Queso untable, fresco y especialidades",
    147: "Arroz y pasta", 148: "Carne", 145: "Fruta y verdura", 154: "Helados", 155: "Hielo", 150: "Marisco", 149: "Pescado", 151: "Pizzas", 884: "Rebozados", 152: "Tartas y churros",
    122: "Atún y otras conservas de pescado", 123: "Berberechos y mejillones", 127: "Conservas de verdura y frutas", 130: "Gazpacho y cremas", 129: "Sopa y caldo", 126: "Tomate",
    201: "Acondicionador y mascarilla", 199: "Champú", 203: "Coloración cabello", 202: "Fijación cabello",
    192: "Afeitado y cuidado para hombre", 189: "Cuidado corporal", 185: "Cuidado e higiene facial", 191: "Depilación", 188: "Desodorante", 187: "Gel y jabón de manos", 186: "Higiene bucal", 190: "Higiene íntima", 194: "Manicura y pedicura", 196: "Perfume y colonia", 198: "Protector solar y aftersun",
    213: "Fitoterapia", 214: "Parafarmacia",
    27: "Fruta", 28: "Lechuga y ensalada preparada", 29: "Verdura",
    77: "Huevos", 72: "Leche y bebidas vegetales", 75: "Mantequilla y margarina",
    226: "Detergente y suavizante ropa", 237: "Estropajo, bayeta y guantes", 241: "Insecticida y ambientador", 234: "Lejía y líquidos fuertes", 235: "Limpiacristales", 233: "Limpiahogar y friegasuelos", 231: "Limpieza baño y WC", 230: "Limpieza cocina", 232: "Limpieza muebles y multiusos", 229: "Limpieza vajilla", 243: "Menaje y conservación de alimentos", 238: "Papel higiénico y celulosa", 239: "Pilas y bolsas de basura", 244: "Utensilios de limpieza y calzado",
    206: "Bases de maquillaje y corrector", 207: "Colorete y polvos", 208: "Labios", 210: "Ojos", 212: "Pinceles y brochas",
    32: "Marisco", 34: "Pescado congelado", 31: "Pescado fresco", 36: "Salazones y ahumados",
    222: "Gato", 221: "Perro", 225: "Otros",
    65: "Bollería de horno", 66: "Bollería envasada", 69: "Harina y preparado repostería", 59: "Pan de horno", 60: "Pan de molde y otras especialidades", 62: "Pan tostado y rallado", 64: "Picos, rosquilletas y picatostes", 68: "Tartas y pasteles", 71: "Velas y decoración",
    897: "Listo para Comer", 138: "Pizzas", 140: "Platos preparados calientes", 142: "Platos preparados fríos",
    105: "Bífidus", 110: "Flan y natillas", 111: "Gelatina y otros postres", 106: "Postres de soja / Yogures griegos", 103: "Yogures desnatados", 108: "Yogures líquidos", 104: "Yogures naturales y sabores", 107: "Yogures y postres infantiles",
    99: "Fruta variada", 100: "Melocotón y piña", 143: "Naranja", 98: "Tomate y otros sabores"
}

# Mapa estructural: Categoría Superior -> Lista de IDs
CATEGORIAS_SUPERIORES = {
    "Aceites, especias y salsas": [112, 115, 116, 117],
    "Agua y refrescos": [156, 163, 158, 159, 161, 162],
    "Aperitivos": [135, 133, 132],
    "Arroz, legumbres y pasta": [118, 121, 120],
    "Azucar, caramelos y chocolate": [89, 95, 92, 97, 90],
    "Bebe": [216, 219, 218, 217],
    "Bodega": [164, 166, 181, 174, 168, 170, 173, 171, 169],
    "Cacao, cafe e infusiones": [86, 81, 83, 84, 88],
    "Carne": [46, 38, 47, 37, 42, 43, 44, 40, 45],
    "Cereales y galletas": [78, 80, 79],
    "Charcuteria y quesos": [48, 52, 49, 51, 50, 58, 54, 56, 53],
    "Congelados": [147, 148, 145, 154, 155, 150, 149, 151, 884, 152],
    "Conservas, caldos y cremas": [122, 123, 127, 130, 129, 126],
    "Cuidado del cabello": [201, 199, 203, 202],
    "Cuidado facial y corporal": [192, 189, 185, 191, 188, 187, 186, 190, 194, 196, 198],
    "Fisioterapia y Parafarmacia": [213, 214],
    "Fruta y verdura": [27, 28, 29],
    "Huevos, leche y Mantequilla": [77, 72, 75],
    "Limpieza y hogar": [226, 237, 241, 234, 235, 233, 231, 230, 232, 229, 243, 238, 239, 244],
    "Maquillaje": [206, 207, 208, 210, 212],
    "Marisco y pescado": [32, 34, 31, 36],
    "Mascotas": [222, 221, 225],
    "Panaderia y Pasteleria": [65, 66, 69, 59, 60, 62, 64, 68, 71],
    "Pizzas y platos preparados": [897, 138, 140, 142],
    "Postres y yogures": [105, 110, 111, 106, 103, 108, 104, 107],
    "Zumos": [99, 100, 143, 98]
}

def scrapear_mercadona_api():
    productos_extraidos = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://tienda.mercadona.es"
    }

    print("🚀 Iniciando extracción masiva del catálogo...\n")

    for cat_superior, lista_ids in CATEGORIAS_SUPERIORES.items():
        print(f"📦 Explorando familia: {cat_superior.upper()}")
        
        for cat_id in lista_ids:
            cat_nombre = SUBCATEGORIAS.get(cat_id, "Desconocida")
            print(f"  📂 Extrayendo subcategoría: {cat_nombre} (ID: {cat_id})")
            
            url = f"https://tienda.mercadona.es/api/categories/{cat_id}/"
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    try:
                        datos = response.json()
                    except ValueError:
                        print(f"  ⚠️ Error en ID {cat_id}: La API no devolvió un JSON válido.")
                        continue
                    
                    # Iterar subcategorías
                    for subcategoria in datos.get("categories", []):
                        for producto in subcategoria.get("products", []):
                            
                            # 1. Nombre principal
                            nombre = producto.get("display_name", "Sin nombre")
                            
                            # 2. Datos de precio y empaquetado
                            price_info = producto.get("price_instructions", {})
                            precio = price_info.get("unit_price", 0.0)
                            
                            # 3. Capacidad y tipo de envase para diferenciar
                            empaque = producto.get("packaging", "") 
                            tamano = price_info.get("unit_size", "")
                            unidad = price_info.get("size_format", "") 
                            formato_completo = f"{empaque} {tamano} {unidad}".strip()
                            
                            thumbnail = producto.get("thumbnail", "")
                            
                            productos_extraidos.append({
                                "Categoria_Superior": cat_superior,
                                "Subcategoria": cat_nombre,
                                "Nombre": nombre,
                                "Formato": formato_completo,
                                "Precio": float(precio),
                                "Imagen": thumbnail
                            })
                else:
                    # Corrección del error para que devuelva el código de estado HTTP real
                    print(f"  ⚠️ Error HTTP {response.status_code} al acceder al ID {cat_id}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Error de conexión en la categoría {cat_id}: {e}")
            
            time.sleep(1.5)
        print("-" * 40)

    if productos_extraidos:
        df = pd.DataFrame(productos_extraidos)
        df.to_csv("catalogo_mercadona.csv", index=False, encoding="utf-8-sig")
        print(f"✅ Extracción completada. {len(productos_extraidos)} productos guardados en 'catalogo_mercadona.csv'")
    else:
        print("❌ No se ha extraído ningún producto.")

if __name__ == "__main__":
    scrapear_mercadona_api()