import cv2
import time
import os

def test_camera_headless():
    print("Kamera tesztelése headless (kijelző nélküli) módban...\n" + "-"*40)
    
    # Próbáljuk meg a 0, 1 és 2-es indexeket (mert USB kamera Linuxon)
    for index in range(3):
        print(f"[{index}] Keresés a /dev/video{index} porton...")
        
        # V4L2 backend használata Linuxhoz
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            print(f"✅ Kamera hardveresen megnyitva a(z) {index}. porton!")
            print("   Kamera bemelegítése (automatikus fényerő beállása)...")
            
            # Olvassunk ki 10 képkockát a "kukába", hogy a kamera fókuszálhasson / beállhasson
            for _ in range(10):
                ret, frame = cap.read()
                time.sleep(0.05)
            
            if ret:
                height, width, channels = frame.shape
                print(f"   Képfelbontás: {width}x{height} (Színcsatornák: {channels})")
                
                # Mentsük el az utolsó leolvasott képet fájlként a nano_client mappába
                filename = f"test_kamera_kep_port_{index}.jpg"
                cv2.imwrite(filename, frame)
                
                print(f"📸 SIKER! Tesztkép elmentve: {filename}")
                print(f"   -> Kattints rá kétszer a VS Code bal oldali sávjában a megtekintéshez!")
                
                cap.release()
                return True
            else:
                print(f"⚠️ A kamera megnyílt, de nem jött át rajta képkocka adata.")
            
            cap.release()
        else:
            print(f"❌ Nincs elérhető eszköz a(z) {index}. porton.")

    print("-" * 40 + "\n❌ Hiba: Nem találtunk működő kamerát. Biztosan be van dugva az USB?")
    return False

if __name__ == "__main__":
    test_camera_headless()