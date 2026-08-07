import dagshub
import mlflow


def init_mlflow():

    dagshub.init(
        repo_owner="adarsh.tiwari.elgoss",
        repo_name="floor-plan-segmentation",
        mlflow=True
    )

    mlflow.set_experiment("YOLO26 Semantic Segmentation")