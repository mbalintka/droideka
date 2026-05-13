import zmq
import cv2
import numpy as np

# ZeroMQ szerver beállítása (PULL mód, ami "szívja" az adatot a Nanoról)
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")

print("📡 Várakozás a Jetson Nano videófolyamára az 5555-ös porton...")

frames_received = 0

try:
    while True:
        # 1. Tömörített JPEG bájtfolyam fogadása a hálózatról
        message = socket.recv()
        
        # 2. Visszafejtés OpenCV képpé
        npimg = np.frombuffer(message, dtype=np.uint8)
        frame = cv2.imdecode(npimg, 1)
        
        if frame is not None:
            frames_received += 1
            # Csak minden 30. képkockánál írunk a terminálba, hogy ne spammeljük tele
            if frames_received % 30 == 0:
                print(f"✅ Sikeresen megérkezett {frames_received} képkocka! Kép felbontása: {frame.shape}")
except KeyboardInterrupt:
    print("\n🛑 Vevő leállítva.")
