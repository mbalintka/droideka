import cv2
import zmq
import time

def start_streaming(camera_index=2, port=5555):
    # 1. ZeroMQ hálózat beállítása (Publisher - Kiadó mód)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # A '*' jelenti, hogy a Nano minden hálózati kártyáján (Wi-Fi, LAN) kiadja az adatot
    socket.bind(f"tcp://*:{port}")
    print(f"📡 ZeroMQ Publisher elindítva a {port}-es porton.")

    # 2. Kamera inicializálása
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"❌ Hiba: Nem sikerült megnyitni a(z) {camera_index}. kamerát.")
        return

    print("🎥 Kamera megnyitva! Streamelés indítása... (Leállítás: Ctrl+C)")
    
    # Képkocka számláló és időzítő a sebesség (FPS) méréséhez
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Nem jött képkocka a kamerából!")
                time.sleep(0.1)
                continue

            # --- ELŐFELDOLGOZÁS A NANO-N ---
            
            # A) Kicsinyítés (pl. 640x480 felbontásra, ha a kamera nagyobb lenne)
            # A DROID-SLAM-nek bőven elég ez a méret, és spórolunk a sávszélességgel.
            frame_resized = cv2.resize(frame, (640, 480))
            
            # B) JPEG Tömörítés (Ez a varázslat!)
            # A minőséget 80%-ra állítjuk. Ez drasztikusan csökkenti a méretet (pl. 1MB -> 40KB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, encoded_image = cv2.imencode('.jpg', frame_resized, encode_param)
            
            if success:
                # 3. Kép elküldése a hálózaton
                # A tobytes() alakítja a képet olyan adathalmazzá (byte-folyammá), amit a hálózat ért is.
                socket.send(encoded_image.tobytes())
                
                frame_count += 1
                
                # Minden 30. képkockánál (kb. másodpercenként) kiírunk egy állapotjelentést
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    size_kb = len(encoded_image.tobytes()) / 1024
                    print(f"Elküldve: {frame_count}. kép | FPS: {fps:.1f} | Méret: {size_kb:.1f} KB")

    except KeyboardInterrupt:
        print("\n🛑 Streamelés leállítva a felhasználó által.")
    finally:
        # Erőforrások felszabadítása a program végén
        cap.release()
        socket.close()
        context.term()
        print("Kamera és hálózat lezárva.")

if __name__ == "__main__":
    # Ha a find_usb_camera.py korábban más indexet adott (pl. 1 vagy 2), azt itt írd át!
    start_streaming(camera_index=2)