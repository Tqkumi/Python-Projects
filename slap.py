import cv2
import numpy as np
import mediapipe as mp
import random
import time

#手の骨格検出モデルへのパス
model_path = r"C:\OIT\media_python\media_python\media\dldata\hand_landmarker.task"

hand_connect = [(0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), 
    (5,9),(9,13),(13,17)]

#各クラス名の省略
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#手検出のモデル，動画モード選択
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)

#手検出器
landmarker = HandLandmarker.create_from_options(options)
#カメラ起動
cap = cv2.VideoCapture(0)

###初期設定###
#状態関数
state = "opening"
#fps
timestamp_ms = 0
#ゲームレベル
level = 1
#手のトラッキング
tracking = False
#手の初期位置
start_x = 0

###ゲーム用###
#ゲーム開始時間
game_start_time = 0
#何秒後によそ見するか
look_away_start = 0
#何秒間よそ見するか
look_away_duration = 0

while True:
    if state == "opening":
        #オープニング画面　背景(黒)
        screen = np.zeros((600, 800, 3), dtype=np.uint8)

        #説明
        cv2.putText(screen, "SLAP GAME",(220,140),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
        cv2.putText(screen, "Max Level:5",(300,200),cv2.FONT_HERSHEY_SIMPLEX,1,(138,44,226),2)
        cv2.putText(screen, "Press S to Start",(260,320),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(screen, "Press Q to Quit",(260,360),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        #オープニング表示
        cv2.imshow("Slap Game", screen)
        cv2.moveWindow("Slap Game", 230, 50)

        key = cv2.waitKey(1)

        if key == ord('s'):
            state = "detect_hand"

        elif key == ord('q'):
            break

    elif state == "detect_hand":
        #カメラ画像取得 retで取得成功判定
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        #BGR→RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #mediapipe用画像へ変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

        #手検出実行
        results = landmarker.detect_for_video(mp_image,timestamp_ms)  
        
        #fps30
        timestamp_ms += 33

        #手を置いておく範囲との境界に線
        cv2.line(frame, (int(w * 0.6), h),(int(w * 0.6), 0), (0,255,0),3)
        cv2.putText(frame,"Put your hand here",(800,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
        cv2.moveWindow("Slap Game", 0, 0)

        #手が見つかったら
        if results.hand_landmarks:
            #検出された複数の手から最初の1つ目の手を取り出す
            hand = results.hand_landmarks[0]
            #手首のx座標
            wrist_x = hand[0].x * w
            #中指のx座標
            middle_fin_x = hand[12].x * w

            #骨格描画
            for start, end in hand_connect:
                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(frame, (x1,y1), (x2,y2), (138,43,226), 2)


            #手が画面の右半分で検知で来たら
            if middle_fin_x > w * 0.6 and wrist_x > w * 0.6:
                cv2.putText(frame,"READY",(400,400),cv2.FONT_HERSHEY_SIMPLEX,5,(0,165,255),5)
                cv2.waitKey(500)
                cv2.imshow("Slap Game", frame)
                cv2.waitKey(500)

                state = "countdown"

        cv2.imshow("Slap Game", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    elif state == "countdown":
        #カウントダウン3秒
        for i in range(3,0,-1):
            screen = np.zeros((600,800,3), dtype=np.uint8)
            #カウントダウン(i秒のstr型)の表示
            cv2.putText(screen,str(i),(350,350),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),5)

            cv2.imshow("Slap Game", screen)
            cv2.waitKey(1000)

        #ゲームタイマー開始
        game_start_time = time.time()
        #よそ見2~7秒のランダムで開始
        look_away_start = random.uniform(2,7)
        #よそ見秒(レベル毎)
        look_away_duration = 1.3 - level*0.2

        tracking = False

        state = "ingame"

    elif state == "ingame":
        ret, frame = cap.read()
        if not ret:
            break

        h,w,_ = frame.shape

        #経過時間
        elapsed = time.time() - game_start_time
        #残り時間
        remain = max(0, 10-int(elapsed))
        #敵がよそ見しているかどうか判定するための変数(経過時間がよそ見開始からよそ見終了までの間にあるかどうか)
        look_away = (look_away_start<= elapsed<= look_away_start + look_away_duration)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        timestamp_ms += 33

        if look_away:
            #よそ見したとき
            cv2.putText(frame,"\\Ouch!/",(270,180),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            cv2.putText(frame,"(>_<)",(300,280),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)

        else:
            #よそ見してないとき
            cv2.putText(frame,"\\Come on!/",(200,180),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
            cv2.putText(frame,"(o_o)",(300,280),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
            cv2.putText(frame,"Aim here!",(250,380),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)

        #ビンタ判定
        if results.hand_landmarks:
            #手首の検出器
            hand = results.hand_landmarks[0]
            #手首位置取得
            middle_x = hand[12].x * w
            for start, end in hand_connect:
                    x1 = int(hand[start].x * w)
                    y1 = int(hand[start].y * h)

                    x2 = int(hand[end].x * w)
                    y2 = int(hand[end].y * h)

                    cv2.line(frame, (x1,y1), (x2,y2), (138,43,226), 2)

            if not tracking:
                #手が画面右側
                if middle_x > w * 0.6:
                    tracking = True
                    start_x = middle_x

            else:
                if middle_x < w * 0.4:
                    #手が画面左側(ビンタ成功)
                    if look_away:
                        cv2.putText(frame,"YOU WIN!",(420,300),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),2)
                        cv2.putText(frame,"NEXT GAME",(400,400),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),2)
                        #画面更新
                        cv2.imshow("Slap Game", frame) 
                        cv2.waitKey(1000)
                        level += 1
                        if(level == 6):
                            cv2.putText(frame,"VICTORY!",(420,300),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,255),2)
                            cv2.imshow("Slap Game", frame) 
                            cv2.waitKey(1000)
                            state = "result"
                        state = "countdown"
                    #ビンタ不成立
                    else:
                        cv2.putText(frame,"YOU LOSE!",(420,300),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
                        cv2.putText(frame,"Too late!",(420,380),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
                        cv2.imshow("Slap Game", frame) 
                        cv2.waitKey(1000)
                        state = "result"

        #レベル表示
        cv2.putText(frame,f"LEVEL: {level}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        #よそ見時間表示
        cv2.putText(frame,f"Binta Chance Time: {look_away_duration}",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        #残り時間表示
        cv2.putText(frame,f"TIME: {remain}",(20,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        #手の置き場所
        cv2.line(frame, (int(w * 0.6), h),(int(w * 0.6), 0), (0,255,0),3)
        #手の案内
        cv2.putText(frame, "Hand is here!",(800,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)

        cv2.imshow("Slap Game", frame)

        #タイムオーバー
        if elapsed >= 10:
            cv2.putText(frame,"YOU LOSE!",(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            state = "result"

        if cv2.waitKey(1) == ord('q'):
            break

    elif state == "result":

        screen = np.zeros((600,800,3), dtype=np.uint8)

        cv2.putText(screen,"GAME OVER...",(200,180),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        cv2.putText(screen,"( ^_^)/~~~<Good Bye~",(200,230),cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)

        cv2.putText(screen,f"Your Max Level : {level}",(280,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.putText(screen,"R : Retry",(300,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.putText(screen,"Q : Quit",(300,470),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        cv2.imshow("Slap Game", screen)

        key = cv2.waitKey(1)

        if key == ord('r'):
            level = 1
            state = "detect_hand"

        elif key == ord('q'):
            break

#全ウィンドウを閉じる
cv2.destroyAllWindows()
cap.release()
landmarker.close()