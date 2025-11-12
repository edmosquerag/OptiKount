import os
import sys
import argparse
import glob
import time
import json
import threading
from collections import Counter # NUEVO: Importamos Counter para contar ocurrencias
from prettytable import PrettyTable
import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import LED

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# --- Carga de Configuracion ---
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

# --- Persistencia de Datos ---
def cargar_datos():
    try:
        with open(config['archivo_datos'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {color: 0 for color in config['colores_a_contar']}

def guardar_datos():
    with open(config['archivo_datos'], 'w') as f:
        json.dump(contador_colores, f, indent=4)

# --- Servidor Flask y SocketIO ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

@app.route('/')
def index():
    # Ahora renderizare tu archivo index.html personalizado
    return render_template('index.html')

@app.route('/datos')
def datos():
    data = dict(contador_colores)
    return jsonify(data)

@app.route('/reset_contador', methods=['POST'])
def reset_contador():
    global contador_colores
    contador_colores = {color: 0 for color in config['colores_a_contar']}
    guardar_datos()
    # MODIFICADO: Cambiado el nombre del evento para que coincida con el HTML
    socketio.emit('update_data', contador_colores)
    return jsonify({"status": "success", "message": "Contadores reiniciados"})


# ---------------------------------------------------------------------------
# --- Configuracion de LEDs y Contadores ---
leds = {color: LED(pin) for color, pin in config['gpio_pins'].items()}
todos = list(leds.values())
contador_colores = cargar_datos()
ultimo_estado_leds = {} # MODIFICADO: Ahora es un diccionario para guardar conteos, no un set

def mostrar_tabla():
    os.system('clear' if os.name == 'posix' else 'cls')
    tabla = PrettyTable()
    tabla.field_names = ["Color", "Detecciones"]
    for color, conteo in contador_colores.items():
        tabla.add_row([color, conteo])
    print(tabla)

# --- FUNCIoN CLAVE CORREGIDA ---
def actualizar_leds(colores_detectados):
    global ultimo_estado_leds, contador_colores

    # 1. Contar las ocurrencias de cada color en el fotograma actual
    # Ej: ['Red', 'Red', 'Blue'] se convierte en {'Red': 2, 'Blue': 1}
    conteo_actual_frame = Counter(colores_detectados)
    
    contador_actualizado = False

    # 2. Iterar sobre los colores detectados en el fotograma actual
    for color, cantidad_actual in conteo_actual_frame.items():
        # Obtenemos cuantos de este color habia en el fotograma anterior (0 si es nuevo)
        cantidad_anterior = ultimo_estado_leds.get(color, 0)
        
        # La diferencia es el numero de NUEVOS objetos de este color
        diferencia = cantidad_actual - cantidad_anterior
        
        if diferencia > 0:
            if color in contador_colores:
                contador_colores[color] += diferencia
                print(f"Detectado {diferencia} nuevo(s) '{color}'. Total: {contador_colores[color]}")
                contador_actualizado = True

    # 3. Actualizar el estado para el proximo fotograma
    ultimo_estado_leds = conteo_actual_frame

    # 4. Si el contador global se actualiza, guardamos y emitimos los datos
    if contador_actualizado:
        guardar_datos()
        socketio.emit('update_data', contador_colores)

    # 5. Logica de LEDs (se mantiene similar, pero usa el nuevo diccionario de conteo)
    for led in todos:
        led.off()

    if conteo_actual_frame.get("Black", 0) > 0:
        for led in todos:
            led.on()
        mostrar_tabla()
        return

    for color in conteo_actual_frame:
        if color in leds and conteo_actual_frame[color] > 0:
            leds[color].on()

    mostrar_tabla()


# --- Argumentos de la linea de comandos ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source', required=True)
parser.add_argument('--thresh', help=f'Minimum confidence threshold (default from config.json: {config["umbral_confianza"]})',
                    default=config['umbral_confianza'])
parser.add_argument('--resolution', help='Resolution in WxH', default=None)
parser.add_argument('--record', help='Record results', action='store_true')

args = parser.parse_args()

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

# --- Servidor Flask en hilo paralelo ---
def iniciar_servidor():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

threading.Thread(target=iniciar_servidor, daemon=True).start()

# --- Bucle Principal de Deteccion ---
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
        if frame_id % 3 != 0:
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
    colores_detectados = []
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
    colores_detectados = [nombre for _, nombre in detecciones_ordenadas]
    actualizar_leds(colores_detectados)

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

# --- Limpieza ---
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
for led in todos:
    led.off()
cv2.destroyAllWindows()
