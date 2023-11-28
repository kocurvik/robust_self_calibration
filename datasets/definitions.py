def get_subset_string(dataset_name):
    if dataset_name == 'haiper':
        save_string = 'im_haiper_{}'
        subsets = ['bike', 'chairs', 'fountain']
        rows, cols = 3, 1

    if dataset_name == 'phototourism':
        save_string = 'im_phototourism_{}'

        # subsets = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
        #            'grand_palace_brussels', 'notre_dame_front_facade', 'palace_of_westminster',
        #            'pantheon_exterior', 'reichstag', 'sacre_coeur', 'st_peters_square',
        #            'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
        subsets = ['buckingham_palace', 'colosseum_exterior',
                   'grand_palace_brussels', 'notre_dame_front_facade', 'palace_of_westminster',
                   'pantheon_exterior', 'reichstag', 'sacre_coeur', 'st_peters_square',
                   'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
        rows, cols = 5, 3

    if dataset_name == 'validation':
        save_string = 'im_phototourism_{}'
        subsets = ['brandenburg_gate']

        rows, cols = 1, 1

    if 'eth' in dataset_name:
        save_string = dataset_name + '_{}'
        subsets = ['delivery_area', 'electro', 'forest', 'playground', 'terrains']
        rows, cols = 3, 2

    if 'multiview' in dataset_name:
        save_string = dataset_name + '_{}'
        subsets = ['statue', 'door', 'boulders', 'botanical_garden', 'lecture_room', 'observatory', 'lounge', 'old_computer', 'bridge', 'terrace_2', 'living_room', 'exhibition_hall']
        rows, cols = 4, 3

    if 'nd_exif' in dataset_name:
        save_string = '{}'
        subsets = ['nd_exif']
        rows, cols = 1, 1

    if 'urban' in dataset_name:
        save_string = 'im_urban_{}'
        subsets = ['kyiv-puppet-theater']
        rows, cols = 1, 1

    if 'aachen' in dataset_name:
        save_string = 'im_aachen_{}'
        subsets = ['aachen_v1.1']
        rows, cols = 1, 1

    if 'single_aachen' == dataset_name:
        save_string = 'single_{}'
        subsets = ['aachen']
        rows, cols = 1, 1

    return save_string, subsets, rows, cols


def img_to_dict(img):
    return {'q': img.qvec, 't': img.tvec}


def cam_to_dict(cam):
    if cam.model in ['PINHOLE', 'THIN_PRISM_FISHEYE']:
        fx = cam.params[0]
        fy = cam.params[1]
        focal = (fx + fy) / 2
        cx = cam.params[2]
        cy = cam.params[3]

        return {'width': cam.width, 'height': cam.height, 'fx': fx, 'fy': fy, 'focal': focal, 'cx': cx, 'cy': cy}

    if cam.model == 'SIMPLE_RADIAL':
        fx = cam.params[0]
        fy = cam.params[0]
        focal = cam.params[0]
        cx = cam.params[1]
        cy = cam.params[2]

        return {'width': cam.width, 'height': cam.height, 'fx': fx, 'fy': fy, 'focal': focal, 'cx': cx, 'cy': cy}

    raise NotImplementedError