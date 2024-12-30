import mlflow
import numpy as np

# Sample input
main_input_data = np.array([
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [2.2, 2.3, 2.4],
        [2.5, 2.6, 2.7],
        [2.8, 2.9, 3.0],
        [3.1, 3.2, 3.3],
        [3.4, 3.5, 3.6],
    ],
    [
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3],
        [1.4, 1.5, 1.6],
        [1.7, 1.8, 1.9],
        [2.0, 2.1, 2.2],
        [2.3, 2.4, 2.5],
        [2.6, 2.7, 2.8],
        [2.9, 3.0, 3.1],
        [3.2, 3.3, 3.4],
        [3.5, 3.6, 3.7],
    ]
], dtype=np.float32)

extra_input_data = main_input_data.copy()
extra_input_data[:, :, 1:] = main_input_data[:, :, 1:] 

combined_input = np.stack((main_input_data, extra_input_data), axis=1)

logged_model = 'runs:/eb346e6d2585451e9d6bb4784a188dea/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

predictions = loaded_model.predict(combined_input)

print(predictions)
