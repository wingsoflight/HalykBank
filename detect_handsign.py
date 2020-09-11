import joblib, argparse, os, fitz, cv2, numpy as np
from utils import preprocess_img, pix2np, get_bounding_rects, get_hist
parser = argparse.ArgumentParser(description='Detect handsigns in pdf doc')
parser.add_argument('path',type=str, nargs=1,
                   help='path to pdf file', metavar='--p')
args = parser.parse_args()
pdf_path = args.path[0]
doc = fitz.open(pdf_path)
clf = joblib.load('SVMcls.pkl')
for i, page in enumerate(doc):
    print('Converting page no. {} to image'.format(i+1))
    zoom = 2  # zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.getPixmap(matrix=mat)
    page_im = pix2np(pixmap)
    prep_im = preprocess_img(page_im)
    rects = get_bounding_rects(prep_im)
    bboxes = []
    print('Detected {} candidate segments.'.format(len(rects)))
    for rect in rects:
        x, y, w, h = rect
        crop = prep_im[y:y+h, x:x+w]
        hist = np.reshape(get_hist(crop), (1, -1))
        pred = clf.predict(hist)
        if pred == 1:
            bboxes.append((x, y, w, h))
    print('Found {} handsigns at page {}'.format(len(bboxes), i+1))
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(page_im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Image', page_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
print('Finished')