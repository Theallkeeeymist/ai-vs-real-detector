# MLflow Guide

## Starting MLflow UI

```bash
mlflow ui
```

Then open: `http://localhost:5000`

## View Experiments

1. Go to http://localhost:5000
2. Click "Experiments" tab
3. Select "ai-vs-real-detector" experiment
4. You'll see all runs with:
   - Model name
   - Hyperparameters
   - Metrics (train/val/test)
   - Artifacts (models, reports, images)

## Compare Runs

1. Select multiple runs (checkboxes)
2. Click "Compare"
3. View side-by-side:
   - Parameters
   - Metrics over time
   - Best metrics

## Model Registry

Access: http://localhost:5000/#/models

Register best model:
```python
from ai_detector.utils.ml_utils.mlflow_utils import MLflowModelRegistry

registry = MLflowModelRegistry()
registry.register_model(
    model_uri="runs:/run_id/model_1",
    model_name="AIVsRealDetector_Model1"
)

# Move to Production
registry.set_model_stage(
    model_name="AIVsRealDetector_Model1",
    version="1",
    stage="Production"
)
```

## Download Artifacts

In MLflow UI:
1. Click on a run
2. "Artifacts" section
3. Download any file/folder
4. Or use API:

```python
client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts(run_id)
```