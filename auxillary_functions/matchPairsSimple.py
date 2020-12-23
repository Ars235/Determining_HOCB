from models.matching import Matching
from models.utils import read_image
from functions import root_dir

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
}

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
matching = Matching(config).eval().to(device)

name0 = root_dir+'/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID1.jpg'
name1 = root_dir+'/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID2.jpg'
rot0, rot1 = 0, 0
# Load the image pair.
image0, inp0, scales0 = read_image(name0, device, [1920, 1920], rot0, False)
image1, inp1, scales1 = read_image(name1, device, [1920, 1920], rot1, False)
# Perform the matching
pred = matching({'image0': inp0, 'image1': inp1})

