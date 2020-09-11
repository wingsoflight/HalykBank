import os, cv2
import numpy as np


def preprocess_img(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.medianBlur(cv2.dilate(img, kernel, iterations=1), 5)
    return img


def preprocess_dir(img_dir):
    img_names = os.listdir(img_dir)
    target_folder = os.path.join(img_dir, 'prep_out')
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = preprocess_img(img_name)
        new_path = os.path.join(target_folder, img_name)
        cv2.imwrite(new_path, img)
    return 0

def segmentation(prep_dir):
    prep_files = os.listdir(prep_dir)
    out_dir = './negative'
    for i, prep_file in enumerate(prep_files):
        img_path = os.path.join(prep_dir, prep_file)
        img = cv2.imread(img_path)
        rects = get_bounding_rects(img)
        for j, rect in enumerate(rects):
            x, y, w, h = rect
            img_path = os.path.join(out_dir, '{}-{}.png'.format(i, j))
            cv2.imwrite(img_path, img[y:y+h, x:x+w])

def get_bounding_rects(img):
    rects = []
    binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.bitwise_not(binary)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    sizes = stats[:, -1]
    for i in range(1, num_labels):
        if sizes[i] < 350:
            continue
        _img = np.where(labels_im==i, 255, 0).astype(np.uint8)
        rects.append(cv2.boundingRect(_img))
    return rects

def convert_to_image(pdf_path):
    import fitz
    filename = os.path.basename(pdf_path)
    out_path = './pdf_out'
    doc = fitz.open(pdf_path)
    zoom = 2  # zoom factor
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        out_file = os.path.join(out_path, '{}-{}.png'.format(filename, i))
        page.getPixmap(matrix=mat).writePNG(out_file)

def get_hist(img):
    binary = cv2.resize(img, (126, 126), interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.bitwise_not(binary)
    gx = cv2.Sobel(binary, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(binary, cv2.CV_64F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    abs_grad_x = cv2.convertScaleAbs(gx)
    abs_grad_y = cv2.convertScaleAbs(gy)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    theta = np.arctan2(gy, gx)
    bins = np.int64(16 * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = [bins[x:x + 9, y:y + 9] for x in range(0, 126, 9) for y in range(0, 126, 9)]
    mag_cells = [mag[x:x + 9, y:y + 9] for x in range(0, 126, 9) for y in range(0, 126, 9)]
    hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
    return np.hstack(hists)

def make_csv():
    import pandas as pd
    data_dir = './prep_out'
    neg_dir = './negative'
    datafiles = os.listdir(data_dir)
    negfiles = os.listdir(neg_dir)
    hists = []
    for filename in datafiles:
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        hists.append(np.append(get_hist(img),1))
    for negfile in negfiles:
        img_path = os.path.join(neg_dir, negfile)
        img = cv2.imread(img_path)
        hists.append(np.append(get_hist(img),0))
    df = pd.DataFrame(hists, index=[i for i in range(len(hists))], columns=['f{}'.format(i) for i in range(3136)]+['sign'])
    df.to_csv('train.csv', index=False)

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im