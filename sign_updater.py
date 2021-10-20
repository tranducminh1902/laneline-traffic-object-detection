import cv2

def update_traffic_sign(label, current_signs):
    if label in [0,1,2,3,4,5,7,8]:
        current_signs['speed'] = label
    elif label == 9:
        current_signs['overtake'] = label
    elif label in [6,32,41]:
        if label == 6 and 'speed' in current_signs:
            del current_signs['speed']
        elif label == 41 and 'overtake' in current_signs:
            del current_signs['overtake']
        else:
            current_signs.clear()
    # elif label in range(33,41):
    #     current_signs['guide'] = label
    return current_signs

def plot_traffic_sign(img_det, current_signs):
    if 'speed' in current_signs:
        speed_label = current_signs['speed']
        speed_sign_img = cv2.imread(f'./trafficsign_meta/{speed_label}.png')
        speed_sign_img = cv2.resize(speed_sign_img,(40,40))
        img_det[420:460, 480:520] = speed_sign_img

    if 'overtake' in current_signs:
        overtake_label = current_signs['overtake']
        overtake_sign_img = cv2.imread(f'./trafficsign_meta/{overtake_label}.png')
        overtake_sign_img = cv2.resize(overtake_sign_img,(40,40))
        img_det[420:460, 530:570] = overtake_sign_img

    if 'guide' in current_signs:
        guide_label = current_signs['guide']
        guide_sign_img = cv2.imread(f'./trafficsign_meta/{guide_label}.png')
        guide_sign_img = cv2.resize(guide_sign_img,(40,40))
        img_det[420:460, 580:620] = guide_sign_img
    return img_det
    