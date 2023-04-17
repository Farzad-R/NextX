# for checking how argparse works: https://docs.python.org/3/howto/argparse.html
# code example:
# - open terminal in the project directory
# - invoke the environment
# - run:

import argparse
from src.clean.CleanWTH import clean_wth
from src.dataprep.PrepareTarget import prepare_target
from src.dataprep.PrepareFeat import prepare_features
from src.dataprep.PrepareTimeFeatures import prepare_time_features
import src.config
import logging

logger = logging.getLogger(__name__)

# assigning the terminal arguments
parser = argparse.ArgumentParser()
parser.add_argument("task", type=str, help="The desired task")


args = parser.parse_args()


def init_tasks():
    clean_tasks = {
        "clean_wth": clean_wth,
        "prepare_target": prepare_target,
        "prepare_features": prepare_features,
        "prepare_time_features": prepare_time_features,
    }

    # Union of dicts
    tasks = {**clean_tasks}
    return tasks


Tasks = init_tasks()


def main(task=args.task):
    try:
        Tasks[task]()
    except:
        logger.error(f"Task {task} does not exist")
        raise


if __name__ == "__main__":
    main()
