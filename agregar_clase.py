import os
import shutil
import sys
from pathlib import Path
 
import kagglehub
 
from config import DATASET_DIR
 
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")
 
 
def explorar(path, max_niveles=2):
    """Muestra la estructura de carpetas para decidir de dónde copiar."""
    base = Path(path)
    print(f"\n📁 Estructura de: {base}\n")
    for carpeta in sorted(base.rglob("*")):
        if carpeta.is_dir():
            profundidad = len(carpeta.relative_to(base).parts)
            if profundidad <= max_niveles:
                n_imgs = len([f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES])
                extra = f"  ({n_imgs} imágenes)" if n_imgs else ""
                print("  " * profundidad + carpeta.name + "/" + extra)
 
 
def copiar_imagenes(origen, destino_clase, n_max):
    dst = Path(DATASET_DIR) / destino_clase
    dst.mkdir(parents=True, exist_ok=True)
 
    origen = Path(origen)
    if not origen.exists():
        print(f"❌ No existe la carpeta: {origen}")
        return
 
    imagenes = [p for p in origen.rglob("*") if p.suffix.lower() in EXTENSIONES][:n_max]
 
    for img in imagenes:
        shutil.copy(img, dst / img.name)
 
    print(f"✅ Copiadas {len(imagenes)} imágenes a {dst}")
 
 
def main():
    if len(sys.argv) < 3:
        print('Uso: python agregar_clase.py "owner/dataset" nombre_clase [carpeta_origen] [n_max]')
        return
 
    slug = sys.argv[1]
    nombre_clase = sys.argv[2]
    carpeta_origen = sys.argv[3] if len(sys.argv) > 3 else None
    n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 500
 
    print(f"Descargando dataset: {slug} ...")
    path = kagglehub.dataset_download(slug)
    print("Descargado en:", path)
 
    if carpeta_origen is None:
        explorar(path)
        print("\n👉 Ahora vuelve a correr el script indicando la carpeta exacta de donde copiar, por ejemplo:")
        print(f'   python agregar_clase.py "{slug}" {nombre_clase} "NOMBRE_DE_CARPETA"')
        return
 
    origen_completo = os.path.join(path, carpeta_origen)
    copiar_imagenes(origen_completo, nombre_clase, n_max)
 
 
if __name__ == "__main__":
    main()