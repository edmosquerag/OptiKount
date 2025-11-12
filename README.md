# OptiKount | Sistema de Inventario Inteligente

Un sistema de inventario en tiempo real que utiliza visión por computadora con YOLO, una Raspberry Pi y una interfaz web para contar objetos de colores.

## Características

- 🎯 **Detección en Tiempo Real**: Utiliza un modelo YOLO para detectar bloques de colores.
- 🌐 **Panel de Control Web**: Interfaz moderna con tabla y gráficos que se actualizan al instante.
- 💡 **Control de Hardware**: Enciende LEDs físicos según los colores detectados.
- 💾 **Persistencia de Datos**: Los conteos se guardan y no se pierden al reiniciar el sistema.
- ⚙️ **Configurable**: Fácil configuración de pines GPIO y otros parámetros vía `config.json`.

## Imágenes del Proyecto


## Requisitos de Hardware

- Raspberry Pi 4 (o similar)
- Cámara USB o PiCamera
- LEDs (5) y resistencias
- Protoboard y cables
- Banda transpotadora

## Requisitos de Software

- Python 3.8+
- Las librerías listadas en `requirements.txt`.

## Instalación

1. **Clona este repositorio:**
    ```bash
    git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
    cd TU_REPOSITORIO
    ```

2. **Crea un entorno virtual (recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3. **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configura el proyecto:**
    - Renombra `config.example.json` a `config.json`.
    - Ajusta los pines GPIO y otros parámetros si es necesario.

5. **Descarga el modelo YOLO:**
    - Coloca tu archivo de modelo (`.pt`) en el directorio raíz del proyecto.

## Uso

Ejecuta el script con la ruta al modelo y la fuente de video:

# Para una cámara USB
python tu_script.py --model=mi_modelo.pt --source=usb0 --resolution=1280x720(`Esta resolucion puede ser ajustada segun se necesite`)

# Para un archivo de video
python tu_script.py --model=mi_modelo.pt --source=video.mp4
