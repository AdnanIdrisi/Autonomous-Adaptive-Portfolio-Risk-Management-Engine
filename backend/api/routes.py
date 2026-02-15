from fastapi import APIRouter
from backend.services.pipeline_service import PipelineService

router = APIRouter()

service = PipelineService()


@router.get("/run-pipeline")
def run_pipeline():
    """
    Runs the full adaptive portfolio engine.
    """
    output = service.run_pipeline()
    return output


@router.get("/metrics")
def get_metrics():
    """
    Returns only performance metrics.
    """
    output = service.run_pipeline()
    return output["performance"]


@router.get("/weights/latest")
def get_latest_weights():
    """
    Returns latest dynamic allocation.
    """
    output = service.run_pipeline()
    return output["latest_weights"]


@router.get("/stress-test")
def get_stress_test():
    """
    Returns stress testing results.
    """
    output = service.run_pipeline()
    return output["stress_test"]
