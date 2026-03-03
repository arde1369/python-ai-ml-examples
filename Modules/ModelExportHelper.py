import datetime, os
import pickle

def export_model_to_file(model, filename):
    saved_models_path="saved_models"
    model_dir = os.path.join(saved_models_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model_path = model_dir + "-" + filename + ".pkl"

    print(f"Saving model to: {model_path}...")

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)