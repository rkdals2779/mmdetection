class DataFrameReaderBase:
    def __init__(self, frame_data, classes):
        self.frame_data = frame_data
        self.classes = classes

    def __len__(self, index):
        return len(self.frame_data)

    def get_image(self, index):
        pass

    def get_box2d(self, index):
        pass

    def get_class(self, index):
        pass

