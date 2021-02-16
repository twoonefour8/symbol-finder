import cv2
import predictor


def findContoursAndSplit(img, predict):
    gray = cv2.GaussianBlur(img, (1, 3), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    im = cv2.filter2D(gray, 0, kernel)

    edged = cv2.Canny(im, 10, 250)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbolsCoordinates = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h < 15 or w > 200:
            continue

        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        letter = im[y:y+h, x:x+w]

        if predict:
            letter = cv2.resize(letter, (64, 64))
            symbol, percent = predictor.predict(letter)
            print(symbol, percent)
            symbolsCoordinates[x] = symbol

    if predict:
        sortedSymbols = [symbolsCoordinates.get(key) for key in sorted(symbolsCoordinates.keys())]
        resultString = ''.join(sortedSymbols)
        print(resultString)

    cv2.imwrite('img/result.jpg', im)


if __name__ == '__main__':
    path = 'img/descript.jpg'
    image = cv2.imread(path)
    doPredict = False
    findContoursAndSplit(image, doPredict)
