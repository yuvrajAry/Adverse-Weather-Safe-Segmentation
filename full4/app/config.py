class AppConfig:
    NUM_CLASSES = 20
    SAFETY_CLASS_IDS = [1, 2, 3, 4]  # example class ids for pedestrians, vehicles, riders, animals
    DEFAULT_INFER_SIZE = (512, 512)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406, 0.5]  # 4th channel for NIR if used
    NORMALIZE_STD = [0.229, 0.224, 0.225, 0.25]
    # Root path to IDDAW dataset on your machine
    DATASET_ROOT = r"D:\iddaw\IDDAW"

