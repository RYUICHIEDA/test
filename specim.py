from time import sleep
from pywinauto import Desktop, Application
import pyautogui
import cv2

def callspecimiq():
    #SpecimIQStudioを動かすための初期設定
    app = Application(backend="uia")
    app.start("C:\Program Files\SpecimIQ\SpecimIQStudio.exe")
    dlg = Desktop(backend="uia")["Specim IQ Studio"]

    #画像を保存するアイコンの画像データ読み込み
    keep = cv2.imread(r'\specimpic\keep.jpg')

    sleep(5)

    #アプリから自動的にカメラと接続
    dlg['DEVICE'].click()
    sleep(1)
    dlg['REMOTE'].click()
    sleep(1)

    #データの保存先の設定
    dlg['CheckBox1'].type_keys("{ENTER}")
    sleep(1)
    dlg['ListItem1'].click_input()
    sleep(1)
    dlg['SELECT Enter'].click()
    sleep(1)
    dlg['START'].click()
    
    sleep(1)

    #撮影
    dlg['SCAN'].click()
    sleep(5)
    dlg['SCAN'].click()

    #画像認識によって撮影終了を検知し、画像を保存する
    maxval = 0
    while True:
        sleep(5)
        scsho=pyautogui.screenshot()
        scsho.save(r'\specimpic\search.jpg')
        scshocv = cv2.imread(r'\specimpic\search.jpg')
        result = cv2.matchTemplate(keep, scshocv, cv2.TM_CCORR_NORMED)
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(result)
        if maxval>0.99:
            pyautogui.click(maxloc[0]+80, maxloc[1]+40, 1)
            break

    sleep(40)

    dlg['Download Now'].click()

    sleep(15)

    dlg['Button7'].click()

    sleep(3)

    app.kill()
    
    return True

#callspecimiq()