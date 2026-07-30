from floor_segmentation.config.configuration import ConfigurationManager
from floor_segmentation.components.prediction import Prediction
from floor_segmentation.utils.polygon_utils import create_prediction_json


class PredictionPipeline:

    def __init__(self):

        config = ConfigurationManager()

        self.prediction_config = config.get_prediction_config()

        self.predictor = Prediction(self.prediction_config)

    def predict(self, image_path):

        results = self.predictor.predict(image_path)

        result = results[0]

        json_output = create_prediction_json(result)

        return json_output