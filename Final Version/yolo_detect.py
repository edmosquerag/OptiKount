import os
import sys
import argparse
import glob
import time
import json
import threading
import base64 # NUEVO: Para codificar la imagen
from collections import Counter
from prettytable import PrettyTable
import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import LED
import psutil # NUEVO: Para obtener información del hardware

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# --- Carga de Configuracion (sin cambios) ---
def cargar_configuracion():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: El archivo 'config.json' no fue encontrado. Asegurate de que existe.")
        sys.exit(0)
    except json.JSONDecodeError:
        print("ERROR: El archivo 'config.json' tiene un formato invalido.")
        sys.exit(0)

config = cargar_configuracion()

# --- Persistencia de Datos (sin cambios) ---
def cargar_datos():
    try:
        with open(config['archivo_datos'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {producto: 0 for producto in config['productos_a_contar']}

def guardar_datos():
    with open(config['archivo_datos'], 'w') as f:
        json.dump(contador_productos, f, indent=4)

# --- Servidor Flask y SocketIO ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*") # NUEVO: cors_allowed_origins para evitar problemas

@app.route('/')
def index():
    return render_template('dashboard.html') # NUEVO: Usaremos un nuevo archivo HTML

@app.route('/datos_iniciales')
def datos_iniciales(): # NUEVO: Endpoint para la carga inicial
    datos = {
        "counts": contador_productos,
        "hardware": obtener_info_hardware()
    }
    return jsonify(datos)

@app.route('/reset_contador', methods=['POST'])
def reset_contador():
    global contador_productos
    contador_productos = {producto: 0 for producto in config['productos_a_contar']}
    guardar_datos()
    # MODIFICADO: Emite el evento completo
    emitir_datos_actualizados()
    return jsonify({"status": "success", "message": "Contadores reiniciados"})

# ---------------------------------------------------------------------------
# --- NUEVO: Funciones para obtener datos del dashboard ---

def obtener_info_hardware():
    """Obtiene la temperatura de la CPU y el uso de la memoria."""
    try:
        temp = psutil.sensors_temperatures()['cpu_thermal'][0].current
    except (KeyError, IndexError):
        temp = 0.0 # Valor por defecto si no se puede leer

    mem = psutil.virtual_memory()
    return {
        "cpu_temp": round(temp, 1),
        "mem_usage": round(mem.percent, 1)
    }

def encontrar_producto_mas_detectado(contador):
    """Encuentra el producto con el conteo más alto."""
    if not any(contador.values()):
        return "Ninguno"
    return max(contador, key=contador.get)

# --- Configuracion de LEDs y Contadores (sin cambios) ---
leds = {producto: LED(pin) for producto, pin in config['gpio_pins'].items()}
todos = list(leds.values())
contador_productos = cargar_datos()
ultimo_estado_detecciones = {}

def mostrar_tabla():
    os.system('clear' if os.name == 'posix' else 'cls')
    tabla = PrettyTable()
    tabla.field_names = ["Producto", "Detecciones"]
    for producto, conteo in contador_productos.items():
        tabla.add_row([producto, conteo])
    print(tabla)

# --- FUNCIÓN CLAVE MODIFICADA ---
def actualizar_conteo_y_leds(productos_detectados):
    global ultimo_estado_detecciones, contador_productos

    conteo_actual_frame = Counter(productos_detectados)
    contador_actualizado = False

    for producto, cantidad_actual in conteo_actual_frame.items():
        cantidad_anterior = ultimo_estado_detecciones.get(producto, 0)
        diferencia = cantidad_actual - cantidad_anterior
       
        if diferencia > 0:
            if producto in contador_productos:
                contador_productos[producto] += diferencia
                print(f"Detectado {diferencia} nuevo(s) '{producto}'. Total: {contador_productos[producto]}")
                contador_actualizado = True

    ultimo_estado_detecciones = conteo_actual_frame

    if contador_actualizado:
        guardar_datos()
        # MODIFICADO: Emite el evento completo en lugar de solo los contadores
        emitir_datos_actualizados()

    for led in todos:
        led.off()

    for producto in conteo_actual_frame:
        if producto in leds and conteo_actual_frame[producto] > 0:
            leds[producto].on()

    mostrar_tabla()

# --- NUEVO: Función principal de emisión de datos ---
def emitir_datos_actualizados():
    """Recopila todos los datos del dashboard y los emite a través de Socket.IO."""
    payload = {
        "counts": contador_productos,
        "fps": round(avg_frame_rate, 1) if 'avg_frame_rate' in globals() else 0,
        "most_detected": encontrar_producto_mas_detectado(contador_productos),
        "hardware": obtener_info_hardware()
    }
    socketio.emit('update_dashboard', payload)


# --- Argumentos de la linea de comandos (sin cambios) ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source', required=True)
parser.add_argument('--thresh', help=f'Minimum confidence threshold (default from config.json: {config["umbral_confianza"]})',
                    default=config['umbral_confianza'])
parser.add_argument('--resolution', help='Resolution in WxH', default=None)
parser.add_argument('--record', help='Record results', action='store_true')

args = parser.parse_args()

# --- Inicialización de cámara y modelo (sin cambios) ---
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
frame_id = 0
video_frame_counter = 0 # NUEVO: Contador para enviar frames de video

# --- Servidor Flask en hilo paralelo (sin cambios) ---
def iniciar_servidor():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

threading.Thread(target=iniciar_servidor, daemon=True).start()

# --- Bucle Principal de Deteccion (con cambios clave) ---
while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % 3 != 0: # Reducir carga de procesamiento
            continue
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Unable to read frames from the Picamera. Exiting program.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    productos_detectados = []
    detecciones_ordenadas = []

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            detecciones_ordenadas.append((xmin, classname))
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    detecciones_ordenadas.sort(key=lambda x: x[0])
    productos_detectados = [nombre for _, nombre in detecciones_ordenadas]
    actualizar_conteo_y_leds(productos_detectados)

    # NUEVO: Emitir el frame de video cada ciertos ciclos para no sobrecargar la red
    video_frame_counter += 1
    if video_frame_counter % 10 == 0: # Enviar 1 de cada 10 frames
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('update_video', frame_b64)

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results', frame)
    if record: recorder.write(frame)

    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)
    
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# --- Limpieza (sin cambios) ---
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
for led in todos:
    led.off()
cv2.destroyAllWindows()
