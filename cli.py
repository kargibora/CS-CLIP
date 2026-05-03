# cli.py
import sys
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from unit_pipeline.main import main


if __name__ == "__main__":
    main()