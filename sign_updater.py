import cv2

def update_traffic_sign(label, current_signs):
    # if label in [0,1,2,3,4,5,7,8]:
    if label in [0,1,3]:
        current_signs['speed'] = label
    elif label in [9,10]:
        current_signs['no_park_stop'] = label
    elif label in [11,12]:
        current_signs['no_left_right'] = label
    # elif label in [13,14]:
    #     current_signs['warn'] = label
    # elif label == 8:
    #   current_signs.clear()
    return current_signs

def plot_traffic_sign(img_det, current_signs):
    if 'speed' in current_signs:
        speed_label = current_signs['speed']
        speed_sign_img = cv2.imread(f'./trafficsign_meta/{speed_label}.png')
        speed_sign_img = cv2.resize(speed_sign_img,(40,40))
        img_det[420:460, 430:470] = speed_sign_img

    if 'no_park_stop' in current_signs:
        no_park_stop_label = current_signs['no_park_stop']
        no_park_stop_sign_img = cv2.imread(f'./trafficsign_meta/{no_park_stop_label}.png')
        no_park_stop_sign_img = cv2.resize(no_park_stop_sign_img,(40,40))
        img_det[420:460, 480:520] = no_park_stop_sign_img

    if 'no_left_right' in current_signs:
        no_left_right_label = current_signs['no_left_right']
        no_left_right_sign_img = cv2.imread(f'./trafficsign_meta/{no_left_right_label}.png')
        no_left_right_sign_img = cv2.resize(no_left_right_sign_img,(40,40))
        img_det[420:460, 530:570] = no_left_right_sign_img

    if 'warn' in current_signs:
        warn_label = current_signs['warn']
        warn_sign_img = cv2.imread(f'./trafficsign_meta/{warn_label}.png')
        warn_sign_img = cv2.resize(warn_sign_img,(40,40))
        img_det[420:460, 580:620] = warn_sign_img
    return img_det
    