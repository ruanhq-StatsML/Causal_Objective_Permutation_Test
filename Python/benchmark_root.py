"""Shared working-directory helper for real-data benchmark scripts."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def chdir_to_script_dir():
    os.chdir(SCRIPT_DIR)
