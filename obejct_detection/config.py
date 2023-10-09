import os

# 현재 모듈의 폴더 획득
curd = os.path.dirname(os.path.abspath(__file__)) + '/'

# 현재 폴더의 하위 dataset 폴더 이미지와 csv파일을 로딩
BASE_PATH = curd + "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "airplanes.csv"])

# 출력 폴더 설정
BASE_OUTPUT = curd + "output"

# 학습 후 저장될 모델 파일 명 설정. 
# 학습 결과 plot.png. 테스트 이미지들.
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# 학습율. 학습 에폭. 배치크기 설정.
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32