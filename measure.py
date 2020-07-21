from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ref_width = 20  # in mm


def read_and_preproces(filename, canny_low= 50, canny_high = 100, blur_kernel=9, d_e_kernel=3):
    # イメージを読む
    image = cv2.imread(filename)
    # グレースケール画像に変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 画像をぼかす
    gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Canny fで適用する
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, (d_e_kernel, d_e_kernel), iterations=1)
    edged = cv2.erode(edged, (d_e_kernel, d_e_kernel), iterations=1)
    return image, edged


image, edged = read_and_preproces('input.JPG')
cv2.imshow("A", edged)
cv2.waitKey()


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)


def get_distance_in_pixels(orig, c):
    # minRect
    box = cv2.minAreaRect(c)
    # MinRectの頂点の座標を取得する
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # Sắp xếp các điểm theo trình tự
    box = perspective.order_points(box)

    # contourをかく
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # エッジの4つの中点を計算する
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # 2次元の長さを計算する
    dc_W = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dc_H = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dc_W, dc_H, tltrX, tltrY, trbrX, trbrY


def find_object_in_pix(orig, edge, area_threshold=3000):
    # 画像でcontourを見つける
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # contourを左から右に配置します
    (cnts, _) = contours.sort_contours(cnts)
    P = None

    # contourを選ぶ
    for c in cnts:
        # 輪郭が小さすぎる場合->スキップ
        if cv2.contourArea(c) < area_threshold:
            continue

        # Pixelでの双方向計算
        dc_W, dc_H, tltrX, tltrY, trbrX, trbrY = get_distance_in_pixels(orig, c)

        # サークルなら
        if P is None:
            # Pをアップデート
            P = ref_width / dc_H
            dr_W = ref_width
            dr_H = ref_width
        else: # 他の形
            dr_W = dc_W * P
            dr_H = dc_H * P

        # 画像に寸法を描く
        cv2.putText(orig, "{:.1f} mm".format(dr_H), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.putText(orig, "{:.1f} mm".format(dr_W), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    return orig


image = find_object_in_pix(image, edged)
cv2.imshow("A", image)
cv2.waitKey()
cv2.destroyAllWindows()
