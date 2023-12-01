import cv2 as cv
import numpy as np
import requests

check_colors = {
    "Красный": np.array([36, 28, 237]),
    "Фиолетовый": np.array([164, 73, 163]),
    "Зеленый": np.array([76, 177, 34]),
    "Синий": np.array([204, 72, 63]),
    "Желтый": np.array([0, 242, 255]),
}

def main():
    url = input()
    resp = requests.get(url, stream=True).raw
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    mx_colors = {
        "Желтый": 0,
        "Зеленый": 0,
        "Красный": 0,
        "Синий": 0,
        "Фиолетовый": 0,
    }

    for color in mx_colors.keys():
        low = check_colors[color]
        high = check_colors[color]
        mask = cv.inRange(img, low, high)
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for S in contours:
            mx_colors[color] = max(mx_colors[color], cv.contourArea(S))

    for color in mx_colors.keys():
        print(color + ": " + str(mx_colors[color]))



if __name__ == '__main__':
    main()