import cv2

def find_working_camera():
    """
    Végigiterál az első 3 videó indexen (0, 1, 2), hogy megtalálja a csatlakoztatott USB kamerát.
    Linux rendszereken a cv2.CAP_V4L2 backend használata javasolt a stabilitás érdekében.
    """
    print(f"OpenCV verzió: {cv2.__version__}")
    print("Kamera keresése indítva...\n" + "-"*30)

    # Próbáljuk meg a 0, 1 és 2-es indexeket
    for index in range(3):
        print(f"[{index}. index] Próbálkozás a /dev/video{index} porton...")
        
        # A cv2.CAP_V4L2 mondja meg az OpenCV-nek, hogy Linuxos USB kamerát keresünk
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # Ha sikerült megnyitni, próbáljunk meg olvasni is belőle egy képkockát
            ret, frame = cap.read()
            if ret:
                print(f"✅ SIKER! Működő kamera azonosítva a(z) {index}. indexen.")
                print(f"   Képfelbontás: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return index
            else:
                print(f"⚠️ A(z) {index}. port megnyílt, de nem jön belőle kép (lehet, hogy csak virtuális eszköz).")
        else:
            print(f"❌ Nem található eszköz a(z) {index}. porton.")
            
        cap.release()

    print("-" * 30 + "\nHiba: Nem találtunk működő kamerát. Kérlek ellenőrizd az USB kábelt!")
    return None

if __name__ == "__main__":
    found_index = find_working_camera()