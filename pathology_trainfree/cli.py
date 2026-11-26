"""Command line for one-shot gland segmentation."""

from .selfref_config import DEFAULT_CONFIG, load_selfref_config


def main(argv=None):
    try:
        cfg = load_selfref_config(argv, default_config=DEFAULT_CONFIG)
        from .selfref_evaluate import run_selfref

        run_selfref(cfg)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from None
