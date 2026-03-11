import cv2
import numpy as np
import time

print("Headless (képernyő nélküli) szimulált kamera indítása...")
print("A leállításhoz nyomd meg a Ctrl+C gombokat a terminálban!\n" + "-"*30)

frame_counter = 0

try:
    while True:
        # 1. Képkocka generálása (ugyanaz a fekete kép a szöveggel)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_counter += 1
        text = f"Frame: {frame_counter}"
        
        # Rárajzoljuk a szöveget a memóriában lévő képre
        cv2.putText(frame, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # 2. Megjelenítés helyett naplózás a terminálba
        # Hogy ne spameljük tele a terminált, csak minden 30. képkockát írjuk ki (másodpercenként egyet)
        if frame_counter % 30 == 0:
            print(f"Sikeresen legenerálva: {text} | Kép felbontása: {frame.shape}")
        
        # 3. Várakozás, hogy tartani tudjuk a kb. 30 FPS sebességet
        time.sleep(1/30)

except KeyboardInterrupt:
    # A Ctrl+C billentyűkombináció megnyomásakor a program szépen kilép
    print("\nFejlesztői leállítás (Ctrl+C) érzékelve. Kamera szimuláció leállítva.")