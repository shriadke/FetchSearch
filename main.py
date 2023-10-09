from fetchSearch.logging import logger
logger.info("This Project uses custom logger.\n")
logger.info("#################################\n")

from fetchSearch.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fetchSearch.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from fetchSearch.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"######## Satge {STAGE_NAME} started. ########")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"######## Satge {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"######## Satge {STAGE_NAME} started. ########")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"######## Satge {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"######## Satge {STAGE_NAME} started. ########")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f"######## Satge {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

    
